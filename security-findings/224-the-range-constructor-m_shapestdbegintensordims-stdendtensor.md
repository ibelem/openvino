# Security finding #224: The range-constructor `m_shape{std::begin(tensor.dims()), std::end(…

**Summary:** The range-constructor `m_shape{std::begin(tensor.dims()), std::end(…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error
**Severity / Impact:** An attacker-supplied ONNX model with negative dimensions can inject SIZE_MAX-valued elements into `m_shape`. Combined with the integer-overflow bypass described below, this causes `ov::op::v0::Constant` (tensor.cpp:494) to be constructed with a shape containing SIZE_MAX dimensions and only a tiny backing buffer. Any downstream inference code that iterates over or allocates based on those dimensions causes heap out-of-bounds read/write or uncontrolled memory allocation (DoS / potential RCE) in any process loading the model.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194` — `Tensor::Tensor(const TensorProto&, ...)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX TensorProto protobuf `dims` field (int64_t) flowing into ov::Shape (vector<size_t>) from model file

## Description / Root cause
The range-constructor `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` performs implicit narrowing conversion of each `int64_t` protobuf dimension to `size_t` with no sign or bounds check. A negative dim value (e.g., -1) silently becomes SIZE_MAX (0xFFFFFFFFFFFFFFFF). The only post-init guard at lines 197-202 only handles the `{0}→scalar` case.

**Validator analysis:** The defect is real. m_shape is an ov::Shape (vector<size_t>); its iterator-range constructor implicitly converts each int64_t protobuf dim to size_t, so a negative dim (e.g. -1) becomes SIZE_MAX with no validation (tensor.hpp:194). The only guard (197-202) only rewrites ov::Shape{0}+size-1 to scalar. In get_ov_constant() (tensor.cpp:438) shape_size({SIZE_MAX,SIZE_MAX}) wraps mod 2^64 to 1, matching a 1-element float_data buffer, so the size-mismatch guard at 467 (1==1) does not fire and ov::op::v0::Constant (tensor.cpp:494) is built with a {SIZE_MAX,SIZE_MAX} shape over a 1-element buffer — a Shape that violates the size_t<->dim invariant and yields uncontrolled allocation / OOB access downstream. CWE-195 (signed→unsigned conversion) is accurate; the impact (OOB read/write, uncontrolled allocation, DoS, potential RCE) is plausible though the most reliable concrete consequence is a corrupted Constant + huge/overflowing downstream allocation. The proposed fix is correct and sufficient: validate `dim >= 0` for every entry before/while building m_shape via FRONT_END_GENERAL_CHECK; the explicit-loop variant is preferable so the cast happens only after validation. Note the validation should reject negatives for ALL dims, not just the demonstrated 2-dim overflow case.

## Exploit / Proof of Concept
Craft an ONNX model with an initializer tensor whose `dims` are [-1, -1] and `float_data` contains exactly 1 element. At tensor.hpp:194, both dims convert to SIZE_MAX. At tensor.cpp:438, `shape_size({SIZE_MAX, SIZE_MAX}) = SIZE_MAX^2 mod 2^64 = 1`. `element_count = 1`. The mismatch guard at line 467 (`1 != 1`) is false → no exception. `ov::op::v0::Constant` is then created with shape `{SIZE_MAX, SIZE_MAX}` over a 1-float buffer. Downstream read of this corrupted constant causes OOB memory access.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195 at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// where `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` narrows
// negative int64_t protobuf dims to SIZE_MAX with no sign check.
//
// Pre-fix: loading a model whose initializer has dims [-1,-1] and a 1-element
//          float_data buffer builds an ov::op::v0::Constant with shape
//          {SIZE_MAX,SIZE_MAX} (size-mismatch guard at tensor.cpp:467 is bypassed
//          because shape_size wraps to 1) -> corrupted Constant / OOB downstream
//          (ASan heap-buffer-overflow on a later read of the constant).
// Post-fix: FRONT_END_GENERAL_CHECK(dim >= 0, ...) throws ov::Exception during
//           convert_model, so the model is rejected cleanly.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// This needs a crafted .onnx fixture, so it is a SKELETON.

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // FrontEndTestUtils::convert_model, manifest macros

using namespace ov::frontend::onnx::tests;

// TODO: provide fixture model 'negative_dims_initializer.onnx' under
//       frontend/onnx/tests/models/ containing an initializer tensor with
//       dims = [-1, -1], data_type = FLOAT, float_data = [1.0f].
TEST(onnx_import_negative_dims, initializer_negative_dims_rejected) {
    // Pre-fix: convert_model succeeds and yields a Constant with SIZE_MAX dims;
    // a subsequent shape/data access trips ASan. Post-fix: throws ov::Exception.
    EXPECT_THROW(convert_model("negative_dims_initializer.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='onnx_import_negative_dims.*'. Provide fixture frontend/onnx/tests/models/negative_dims_initializer.onnx (initializer dims=[-1,-1], FLOAT, float_data=[1.0]). Pre-fix expected (ASan build): heap-buffer-overflow / allocation-size-too-big when the {SIZE_MAX,SIZE_MAX} Constant is materialized/read; post-fix: ov::Exception 'negative dimension' during convert_model, test passes.

## Suggested fix
Add a sign-validation loop before or during construction of `m_shape`. For example, immediately before line 194: `for (auto dim : tensor.dims()) { FRONT_END_GENERAL_CHECK(dim >= 0, "Tensor initializer has negative dimension: ", dim); }`. Alternatively, replace the range constructor with an explicit loop that checks `dim >= 0` and casts only after validation: `for (auto dim : tensor.dims()) { FRONT_END_GENERAL_CHECK(dim >= 0, ...); m_shape.push_back(static_cast<size_t>(dim)); }`. This ensures no negative value ever reaches `m_shape`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #224.
