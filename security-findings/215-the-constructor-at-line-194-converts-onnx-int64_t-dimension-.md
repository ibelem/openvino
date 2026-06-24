# Security finding #215: The constructor at line 194 converts ONNX `int64_t` dimension value…

**Summary:** The constructor at line 194 converts ONNX `int64_t` dimension value…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Any ONNX tensor with a negative dimension (including -1, commonly used for dynamic shapes in other frameworks) is silently accepted and stored as SIZE_MAX in `ov::Shape`. Downstream consumers of the shape — including `shape_size()`, `Constant` constructors, and memory allocators — receive this invalid value without warning, risking DoS (excessive allocation), OOB read/write, or other memory-safety violations.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194` — `Tensor::Tensor(const TensorProto&, ...)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file → Tensor constructor: `tensor.dims()` are attacker-controlled int64_t values

## Description / Root cause
The constructor at line 194 converts ONNX `int64_t` dimension values directly to `ov::Shape` (which is `std::vector<size_t>`) via `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` with no validation that dimensions are non-negative. A negative int64_t dim (e.g., -1 = 0xFFFFFFFFFFFFFFFF) is silently truncated/reinterpreted as SIZE_MAX, producing a semantically invalid enormous shape. This is an enabler for the CWE-191 bypass in `get_ov_constant()` (finding above) and may independently cause integer overflow in `shape_size()` for multi-dimensional shapes.

**Validator analysis:** The cited flaw is real: tensor.hpp:191-202 builds ov::Shape (vector<size_t>) directly from int64 ONNX dims with no validation, so negative dims silently become SIZE_MAX. CWE-20 is the accurate categorization. For the single-dimension case the impact claim is overstated — the size-mismatch check at tensor.cpp:467 throws error::invalid_external_data before any allocation, so a lone dims:[-1] is rejected gracefully (no DoS/OOB). However the validation gap is genuinely exploitable via the multi-dimensional overflow noted in the finding: dims:[-1,-1] yields shape_size = (2^64-1)^2 mod 2^64 = 1, which matches a 1-element initializer, bypasses the line-467 guard, and constructs an ov::Constant carrying a semantically invalid {SIZE_MAX,SIZE_MAX} shape backed by a 1-element buffer — a malformed tensor that propagates into the graph and can drive downstream shape-inference/allocation misbehavior. The proposed fix (iterate tensor.dims() and FRONT_END_GENERAL_CHECK(d >= 0)) is correct and sufficient: rejecting negative dims at construction blocks both the SIZE_MAX truncation and the wraparound bypass. Recommend placing the check inside the constructor body (after m_shape init or before) so it fires for every initializer. Triggering requires a crafted .onnx with a negative initializer dim, so the regression test needs a binary fixture rather than a pure-API exercise.

## Exploit / Proof of Concept
Supply an ONNX model with `dims: [-1]` on any tensor initializer. The Tensor constructor stores `ov::Shape{SIZE_MAX}`. Any code that subsequently calls `shape_size(m_shape)` or allocates storage for this shape will attempt to handle a SIZE_MAX-element tensor. Combined with the nibble underflow (finding 1), this completely silences the mismatch check.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-20 in openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// The Tensor(TensorProto&,...) ctor builds ov::Shape from int64 dims with no non-negative
// validation, so a negative dim becomes SIZE_MAX and dims:[-1,-1] overflow-wraps shape_size to 1,
// bypassing the size-mismatch guard in tensor.cpp:467. After the fix the model must be rejected.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp (ov_onnx_frontend_tests, gtest).
// TODO: add a binary fixture models/negative_dim_initializer.onnx whose initializer has
//       dims:[-1,-1] (or dims:[-1]) and matching raw_data — there is no pure-API way to inject a
//       negative protobuf dim through the C++ frontend API, so a crafted .onnx is required.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// Expected to FAIL pre-fix (model loads / Constant built with SIZE_MAX shape, possibly ASan abort
// downstream); PASSES post-fix because the ctor throws on the negative dimension.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_initializer_dimension_rejected) {
    EXPECT_THROW(convert_model("negative_dim_initializer.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_model_negative_initializer_dimension_rejected* (built with -DENABLE_SANITIZER=ON). Provide fixture src/frontends/onnx/tests/models/negative_dim_initializer.onnx with an initializer dims:[-1,-1] and a 1-element raw_data. Pre-fix: convert_model does NOT throw (the size-mismatch guard is bypassed by the shape_size wraparound) and the resulting Constant carries a {SIZE_MAX,SIZE_MAX} shape; ASan/UBSan may flag an out-of-bounds or huge allocation in downstream shape handling. Post-fix: the Tensor ctor's FRONT_END_GENERAL_CHECK(d >= 0) throws ov::Exception, so EXPECT_THROW passes.

## Suggested fix
Add explicit validation in the Tensor constructor (tensor.hpp:191–203): after constructing `m_shape`, iterate over `tensor.dims()` and throw `error::invalid_model` (or use `OPENVINO_ASSERT`) if any dim value is negative. E.g.: `for (auto d : tensor.dims()) { FRONT_END_GENERAL_CHECK(d >= 0, "Tensor '" + tensor.name() + "' has negative dimension ", d); }`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #215.
