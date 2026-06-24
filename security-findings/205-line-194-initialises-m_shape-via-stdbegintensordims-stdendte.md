# Security finding #205: Line 194 initialises `m_shape` via `{std::begin(tensor.dims()), std…

**Summary:** Line 194 initialises `m_shape` via `{std::begin(tensor.dims()), std…

**CWE IDs:** CWE-194: Unexpected Sign Extension / CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** An attacker who supplies a crafted ONNX initializer tensor with a dim of -1 causes `m_shape` to contain SIZE_MAX. Any downstream shape-arithmetic that multiplies or sums dims (e.g. total element count for buffer allocation) will integer-overflow. If the resulting count is still 'small', a heap allocation sized by it will be far smaller than the code believes, leading to a heap out-of-bounds write when the tensor data is copied; if the count is huge, it causes an allocation failure / denial of service.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194` — `Tensor::Tensor(const TensorProto&, const std::filesystem::path&, detail::MappedMemoryHandles)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX TensorProto.dims (int64_t, attacker-controlled from untrusted model file) → ov::Shape (std::vector<size_t>)

## Description / Root cause
Line 194 initialises `m_shape` via `{std::begin(tensor.dims()), std::end(tensor.dims())}`. `tensor.dims()` yields `int64_t` values; `ov::Shape`'s range constructor at `shape.hpp:38` passes them directly to `std::vector<size_t>(first, last)` — the implicit signed-to-unsigned conversion silently wraps any negative dim (e.g. -1) to SIZE_MAX (0xFFFFFFFFFFFFFFFF). The only post-construction check (lines 197-202) fires only when `m_shape == ov::Shape{0}` and does not iterate dims to detect negative-turned-huge values.

**Validator analysis:** Confirmed at tensor.hpp:194 — m_shape{std::begin(tensor.dims()), std::end(tensor.dims())} performs an implicit int64_t→size_t conversion in the std::vector<size_t> range ctor. A negative dim (e.g. -1) silently becomes SIZE_MAX; the post-ctor check at 197-202 only normalizes ov::Shape{0} scalars and never iterates dims to reject negatives. tensor.dims() is attacker-controlled (untrusted ONNX initializer). CWE-194/CWE-190 categorization is accurate. The impact wording is partially overstated: a single -1 dim yields shape_size==SIZE_MAX which most plausibly causes a std::bad_alloc / OOM DoS rather than a tidy small-allocation heap overflow; the small-allocation/heap-OOB-write scenario requires a multi-dim product that wraps to a small value (achievable with crafted multiple negative/large dims), so the OOB-write impact is reachable but conditional. Either way a real defect with at-minimum DoS, at-worst heap corruption. The proposed fix (FRONT_END_GENERAL_CHECK(d >= 0, ...) loop before line 194) is correct and sufficient for the sign-wrap; it would be slightly stronger to also reject dims whose product exceeds a sane bound to fully close the multiplication-overflow path, but the non-negative check alone removes the SIZE_MAX wrap and is the necessary fix. Caveat for openvinoEp: ORT may pre-validate concrete initializer dims, but the EP's serialize-subgraph→OpenVINO-frontend compile path can still drive untrusted dims into this ctor, so it remains reachable.

## Exploit / Proof of Concept
Craft an ONNX model with an initializer whose `dims` field contains a single entry of value -1 (e.g. `tensor.dims: [-1]`). When `Tensor::Tensor` is called, line 194 stores SIZE_MAX into `m_shape[0]`. Any caller that then computes `ov::shape_size(m_shape)` (product of all dims) gets 0xFFFFFFFFFFFFFFFF or an arithmetic wrap, and a subsequent `std::vector<T>` or `AlignedBuffer` allocation uses that as element count, enabling a heap buffer overflow or OOM crash.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes fix for CWE-194/CWE-190 at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// where ov::Shape is constructed from int64_t TensorProto.dims() with no
// non-negative check, so a dim of -1 wraps to SIZE_MAX.
//
// Pre-fix: convert_model on a model whose initializer has dims:[-1] either
//   builds a Constant with a SIZE_MAX dimension (huge shape_size -> bad_alloc/
//   abort under ASan) or silently produces a corrupt shape.
// Post-fix: the frontend throws ov::Exception (FRONT_END_GENERAL_CHECK)
//   rejecting the negative dim, which this test asserts.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: requires a crafted binary fixture (negative_initializer_dim.onnx /
//       .prototxt) with an initializer tensor declaring dims:[-1]. The .onnx
//       must be added under the frontend test models dir and referenced via
//       the test's util::path helper — see TODO below.

#include "onnx_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/negative_initializer_dim.prototxt containing an
//       initializer (e.g. a FLOAT tensor named "x") whose dims field is [-1].
//       Use convert_partially/convert_model exactly as other onnx_import tests do.
TEST(onnx_import_negative_dim, initializer_negative_dim_is_rejected) {
    EXPECT_THROW(convert_model("negative_initializer_dim.onnx"), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_import_negative_dim.* . Pre-fix expectation: no exception thrown; instead ASan/abort from std::bad_alloc or a heap overflow when allocating a Constant sized by the SIZE_MAX-wrapped shape (so EXPECT_THROW fails / process aborts). Post-fix: convert_model throws ov::Exception ("Tensor dim must be non-negative") and the test passes. TODO: supply the crafted negative_initializer_dim.onnx fixture before this compiles/runs.

## Suggested fix
Before line 194, validate every dim is non-negative. Example: `for (auto d : tensor.dims()) { FRONT_END_GENERAL_CHECK(d >= 0, "Tensor dim must be non-negative, got ", d); }` — place this check immediately before the `m_shape` initialisation so that a crafted negative dim throws a descriptive frontend error rather than silently wrapping.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #205.
