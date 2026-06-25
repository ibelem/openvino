# Security finding #464: Line 194: `m_shape{std::begin(tensor.dims()), std::end(tensor.dims(…

**Summary:** Line 194: `m_shape{std::begin(tensor.dims()), std::end(tensor.dims(…

**CWE IDs:** CWE-194: Unexpected Sign Extension / CWE-20: Improper Input Validation
**Severity / Impact:** A single negative dim in the ONNX initializer produces a SIZE_MAX-magnitude shape dimension. Combined with the shape_size overflow (finding 1), this bypasses the element-count consistency check and creates a Constant with a nonsensical shape. Downstream inference operators that iterate or index by shape will read/write out-of-bounds memory. Affects all users loading untrusted ONNX models.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194` — `Tensor::Tensor(const TensorProto&, ...)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file: TensorProto.dims field (repeated int64) copied directly into ov::Shape (vector<size_t>) with no sign or range check

## Description / Root cause
Line 194: `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` silently converts each `int64` dimension from the ONNX proto into `size_t`. Negative values (e.g. dims=-1 as int64 = 0xFFFFFFFFFFFFFFFF) are silently reinterpreted as enormous `size_t` values via implicit signed-to-unsigned conversion. There is no check that dims values are non-negative. This creates an ov::Shape containing SIZE_MAX-like values which then feeds both the `shape_size` overflow and direct use of the giant Shape when constructing ov::op::v0::Constant downstream.

**Validator analysis:** The cited line 194 (`m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}`) does perform an unchecked int64->size_t conversion; the only dims-related guard is the dims==[0] scalar edge-case at line 197, which does NOT reject negatives. A dim of -1 (0xFFFFFFFFFFFFFFFF) becomes size_t SIZE_MAX. CWE-194/CWE-20 are accurate categorisations. The exploit's bypass of the element-count check in get_ov_constant (tensor.cpp:467) is plausible: with dims=[-1,-1], shape_size(m_shape) wraps to 1 (since (2^64-1)^2 mod 2^64 == 1), so supplying exactly one element passes line 467 and a Constant is built at tensor.cpp:494/480 with a nonsensical {SIZE_MAX,SIZE_MAX} shape but tiny backing storage. No immediate OOB occurs at Constant construction (it copies shape_size==1 element), so the corruption manifests downstream where ops iterate by the corrupt shape — the impact is therefore real but indirect, contingent on downstream shape consumers, somewhat less certain than 'guaranteed OOB'. The proposed fix (FRONT_END_GENERAL_CHECK(d >= 0,...) before constructing m_shape) is correct and sufficient for the sign-extension flaw itself; it should additionally cap the per-dim and total product to a sane bound to also defang the shape_size overflow (the related finding). Validating non-negativity alone removes the SIZE_MAX values that drive the wraparound.

## Exploit / Proof of Concept
Set TensorProto.dims = [-1] (a single dimension of -1, stored as int64 0xFFFFFFFFFFFFFFFF). Tensor constructor at line 194 casts it to size_t, giving m_shape = {18446744073709551615}. shape_size returns 18446744073709551615, but get_data_size() returns 0 if no data is provided, so both are 0 only if additional wraparound occurs. More directly: dims=[-1, -1] gives shape_size = SIZE_MAX*SIZE_MAX mod 2^64 = 1. Supply exactly 1 float element as raw_data. The check passes, and a Constant with shape {SIZE_MAX, SIZE_MAX} backed by 4 bytes is created.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing sign/range check at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// (Tensor::Tensor: int64 dims copied into ov::Shape<size_t> with no >=0 check).
//
// Pre-fix: a TensorProto initializer with dims=[-1,-1] sign-extends to
// m_shape={SIZE_MAX,SIZE_MAX}; shape_size wraps to 1 and the model loads,
// producing a Constant with a corrupt shape (no exception thrown).
// Post-fix: the frontend must reject the negative dim and throw ov::Exception.
//
// This is a SKELETON: it requires a crafted ONNX model containing an initializer
// whose TensorProto.dims = [-1,-1] with a single matching data element. That
// binary fixture cannot be authored inline here, hence the TODO below.
//
// Style follows onnx_import.in.cpp (ov_onnx_frontend_tests).
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: add models/negative_initializer_dim.onnx to the onnx test models dir.
//       It must contain one initializer with TensorProto.dims = [-1, -1]
//       (int64 0xFFFFFFFFFFFFFFFF each) and a single float element of data,
//       so that the pre-fix shape_size wraparound (==1) passes tensor.cpp:467.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_initializer_negative_dim_is_rejected) {
    EXPECT_THROW(convert_model("negative_initializer_dim.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON for ASan coverage of downstream shape use). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_model_initializer_negative_dim_is_rejected*'. Pre-fix the test FAILS because convert_model returns a model (no throw) for the sign-extended {SIZE_MAX,SIZE_MAX} shape (and ASan may later report a heap-buffer-overflow / allocation-size-too-big if the corrupt-shape Constant is consumed); post-fix it PASSES because the new FRONT_END_GENERAL_CHECK(d >= 0) at tensor.hpp:194 throws ov::Exception. Requires the crafted negative_initializer_dim.onnx fixture (see TODO).

## Suggested fix
In the Tensor constructor (tensor.hpp:194), before constructing m_shape, iterate over `tensor.dims()` and reject any value ≤ 0 (except for the documented scalar edge-case where `dims=[0]` and data_size=1) or any value exceeding a safe maximum (e.g. 1<<30 per dimension, or total product > 2^33). Throw `error::invalid_model` or `FRONT_END_GENERAL_CHECK` failure. Example: `for (auto d : tensor.dims()) { FRONT_END_GENERAL_CHECK(d >= 0, "Tensor dim must be non-negative"); }` placed before constructing m_shape.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #464.
