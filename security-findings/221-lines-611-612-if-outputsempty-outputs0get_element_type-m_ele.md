# Security finding #221: Lines 611-612: `if (!outputs.empty() && outputs[0].get_element_type…

**Summary:** Lines 611-612: `if (!outputs.empty() && outputs[0].get_element_type…

**CWE IDs:** CWE-805: Buffer Access with Incorrect Length Value
**Severity / Impact:** Same as the evaluate() finding above — OOB heap read, information disclosure, potential crash. This function is an explicit triggering path for the evaluate() flaw: the type-mismatch condition is checked and then the mismatched tensor is forwarded verbatim.
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:611` — `Constant::evaluate_lower()`
**Validated for repos:** openvino
**Trust boundary:** evaluate_lower() accepts an externally-provided TensorVector whose element type is compared against m_element_type; when mismatched it delegates to evaluate() without type normalization.

## Description / Root cause
Lines 611-612: `if (!outputs.empty() && outputs[0].get_element_type() != m_element_type) return evaluate(outputs, {});` forwards a type-mismatched output tensor directly into evaluate(), which at line 599 performs memcpy sized by the output tensor's byte size. No conversion or size clamping is done before the delegation, so the wider-type output tensor causes an over-read in evaluate().

**Validator analysis:** Confirmed real in openvino core. evaluate() (constant.cpp:586-603) sets outputs[0] shape to m_shape but keeps the caller-supplied element type, then memcpy copies outputs[0].get_byte_size() bytes from get_data_ptr(). The source buffer m_data is sized for m_element_type; if the output element type is wider (e.g. f32 vs stored f16) get_byte_size() exceeds the source size and memcpy over-reads by the type-width ratio. evaluate_lower() (611-612) and evaluate_upper() (619-620) explicitly take the type-mismatch branch and forward the wider tensor verbatim, with no conversion or clamp — exactly the cited CWE-805 over-read. vulnType (CWE-805 / OOB heap read) and impact (info disclosure / crash) are accurate; note it only over-reads when the dst type is *wider* than m_element_type (a narrower dst merely truncates). The proposed fix is correct: a same-type raw-copy fast path plus a separate element-wise conversion path (allocate a correctly-typed temp, evaluate, then convert into outputs[0]) is the right shape; an even simpler hardening is to clamp the memcpy length to std::min(outputs[0].get_byte_size(), shape_size(m_shape)*m_element_type.size()) and assert on type mismatch, but real correctness for TypeRelaxed needs the conversion, so option (b) is preferred. Reachability is demonstrated within openvino's public Constant API / TypeRelaxed bound-evaluation; it is not demonstrably reachable through the EP's ONNX-import boundary, hence openvinoEp rejected.

## Exploit / Proof of Concept
Call evaluate_lower() on a TypeRelaxed<Constant<f16>> with an outputs[0] of type f32 (same shape). The condition at line 611 is true, evaluate() is called, and the memcpy at line 599 over-reads the f16 m_data buffer by a factor of 2.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
#include <gtest/gtest.h>

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

// Encodes the fix for the OOB heap read in Constant evaluation.
// Source of defect:
//   src/core/src/op/constant.cpp:599  -> memcpy(outputs[0].data(), get_data_ptr(), outputs[0].get_byte_size())
//   src/core/src/op/constant.cpp:611-612 -> evaluate_lower() forwards a type-mismatched
//                                            (wider) output tensor straight into evaluate().
// Pre-fix behaviour: a same-shape f32 (4-byte) output tensor causes evaluate() to copy
//   shape_size*4 bytes out of the f16 (2-byte) source buffer -> 2x over-read.
//   Under ASan this aborts with "heap-buffer-overflow READ" inside Constant::evaluate.
// Post-fix behaviour: evaluate_lower() must convert (or reject) the mismatched type
//   instead of raw-copying, so no OOB read occurs.
TEST(constant_evaluate_lower, type_mismatch_no_oob_read) {
    using namespace ov;

    const Shape shape{4};
    std::vector<float16> src(shape_size(shape), float16(1.0f));
    auto c = std::make_shared<op::v0::Constant>(element::f16, shape, src.data());

    // Wider-typed output tensor of the same shape, emulating TypeRelaxed<Constant>.
    Tensor wider_out(element::f32, shape);
    TensorVector outputs{wider_out};

    // Pre-fix: ASan flags an OOB read during the memcpy in Constant::evaluate.
    // Post-fix: completes safely (value converted / mismatch handled).
    ASSERT_NO_THROW(c->evaluate_lower(outputs));
    // TODO: once the fix lands, also assert the produced values equal the
    //       f16->f32 converted source (not a raw bit reinterpretation).
}
```
**Build / run:** Build the core unit-test target (e.g. cmake --build . --target ov_core_unit_tests; confirm exact name from src/core/tests/ CMake). Run: ./ov_core_unit_tests --gtest_filter=constant_evaluate_lower.type_mismatch_no_oob_read with an ASan-instrumented build. Pre-fix expected failure: AddressSanitizer 'heap-buffer-overflow READ of size N' originating from std::memcpy in ov::op::v0::Constant::evaluate (src/core/src/op/constant.cpp:599) reached via evaluate_lower (constant.cpp:611-612). Post-fix: test passes with no ASan report.

## Suggested fix
Do not forward a type-mismatched output tensor to evaluate(). Instead, either: (a) perform an element-wise type conversion in evaluate_lower() itself; or (b) create a temporary correctly-typed tensor, call evaluate() on it, then convert the result into outputs[0]. This separates the raw-copy fast path (same type) from the conversion path (different type).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #221.
