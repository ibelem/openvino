# Security finding #463: At line 438, `shape_size(m_shape)` is called where `m_shape` contai…

**Summary:** At line 438, `shape_size(m_shape)` is called where `m_shape` contai…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Heap out-of-bounds read/write during inference. Any downstream op that indexes the Constant using the declared (giant) shape will access memory far beyond the small buffer. Under a malicious ONNX model this allows denial-of-service (crash) or, with layout knowledge, arbitrary memory read/write and potential RCE. Any user loading an untrusted ONNX model through the OpenVINO ONNX frontend is affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:438` — `Tensor::get_ov_constant()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model file: TensorProto.dims int64 fields ingested via Graph::Graph → Tensor{initializer_tensor, model_dir, mmap_cache} → get_ov_constant()

## Description / Root cause
At line 438, `shape_size(m_shape)` is called where `m_shape` contains `size_t` values copied verbatim from attacker-controlled `tensor.dims()` (int64). The `shape_size` template at `core/include/openvino/core/shape.hpp:92-95` uses `std::accumulate` with `std::multiplies<size_t>` — pure wrapping unsigned multiplication with no overflow guard. The wrapped (small) result is stored as `shape_elements` and compared against `element_count` at line 467. If the attacker crafts dims such that the product wraps to a small value (e.g. dims=[2, 0x8000000000000001] → product=2), and supplies exactly that many inline data bytes, the check at line 467 (`element_count != shape_elements`) passes silently. A `ov::op::v0::Constant` is then constructed at line 494 with the giant `m_shape` but only a tiny backing buffer.

**Validator analysis:** The flaw is real. tensor.hpp:194 copies attacker-controlled int64 TensorProto.dims directly into ov::Shape (size_t) with no validation; negative/huge dims become large size_t values. shape_size at tensor.cpp:438 delegates to shape.hpp:92-95 which is plain unsigned accumulate/multiplies — it wraps modulo 2^64 with no overflow check. A crafted dims=[2, 0x8000000000000001] wraps to product=2, matching element_count derived from 8 bytes of inline data, so the consistency check at line 467 is silently bypassed and a Constant with a giant declared shape but tiny backing buffer is built at line 494. CWE-190 is the correct classification. Impact: the practical, certain consequence is OOB read / DoS when downstream ops iterate the constant by its declared (non-wrapped) dimensions or compute strides; the 'arbitrary RCE' claim is speculative and should be downgraded to OOB/DoS. The proposed fix is correct and sufficient: (1) reject negative/absurd dims in the Tensor constructor (tensor.hpp:194), and (2) replace bare shape_size with an overflow-checked product (r<=SIZE_MAX/d before each multiply). I'd add that the same overflow class affects any other shape_size consumer, so the constructor-side dimension validation is the more robust mitigation. Both repo verdicts are validated: the defect physically lives in openvino/src and is reachable from the OpenVINO EP boundary since the EP loads untrusted ONNX through OpenVINO's ONNX frontend.

## Exploit / Proof of Concept
Craft an ONNX TensorProto with data_type=FLOAT, dims=[2, 0x8000000000000001], and raw_data of 8 bytes (2 float32 values). On 64-bit: shape_size({2, 9223372036854775809}) = 2*9223372036854775809 mod 2^64 = 2, matching element_count=2, so line 467 does not throw. make_shared<ov::op::v0::Constant>(float32, {2,9223372036854775809}, get_data_ptr()) is called at line 494 with a pointer to 8 bytes but a declared shape of ~18 EB. Any inference op that walks the Constant data using the declared shape will read/write far past the 8-byte buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 in Tensor::get_ov_constant
//   targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:438 (shape_size wrap)
//   targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194 (unvalidated int64 dims -> size_t)
//
// Pre-fix: dims=[2, 0x8000000000000001] wrap to shape product 2, matching 8 bytes of
// inline FLOAT raw_data, so the size check at tensor.cpp:467 does NOT throw and a
// Constant with a giant declared shape over an 8-byte buffer is created; downstream
// indexing reads OOB (ASan heap-buffer-overflow). Post-fix: the frontend must reject
// the overflowing/absurd dimension and throw ov::Exception during convert_model.
//
// This needs a crafted binary .onnx fixture, so it is a SKELETON.

#include "onnx_utils.hpp"          // FrontEndTestUtils / convert_model helpers used by onnx_import.in.cpp
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: create the fixture model under
//   src/frontends/onnx/tests/models/tensor_dims_overflow.onnx
// with an initializer: data_type=FLOAT, dims=[2, 0x8000000000000001],
// raw_data = 8 bytes (two float32 values). On 64-bit the size_t product wraps to 2.
TEST(onnx_importer, DISABLED_tensor_dims_integer_overflow_rejected) {
    // TODO: replace with the project's model-path helper (see onnx_import.in.cpp).
    EXPECT_THROW(convert_model("tensor_dims_overflow.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Add fixture src/frontends/onnx/tests/models/tensor_dims_overflow.onnx (FLOAT, dims=[2,0x8000000000000001], 8-byte raw_data). Run: ov_onnx_frontend_tests --gtest_filter=onnx_importer.*tensor_dims_integer_overflow* (remove DISABLED_ once fixture exists). Pre-fix expectation: ASan heap-buffer-overflow (or no throw) when the Constant is consumed; post-fix: convert_model throws ov::Exception (invalid_model) and the test passes.

## Suggested fix
Before calling `shape_size(m_shape)` at line 438, validate every dimension: (1) In the Tensor constructor (tensor.hpp:194), reject any `tensor.dims()` value that is negative or exceeds a reasonable maximum (e.g. > 1<<30), throwing an `error::invalid_model` exception. (2) Replace the bare `shape_size` call with an overflow-safe product: iterate and check `current_product > SIZE_MAX / dim` before each multiply, throwing on overflow. Example: `size_t safe_shape_size(const ov::Shape& s) { size_t r=1; for (auto d:s){ FRONT_END_GENERAL_CHECK(d==0||r<=SIZE_MAX/d, "shape overflow"); r*=d; } return r; }`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #463.
