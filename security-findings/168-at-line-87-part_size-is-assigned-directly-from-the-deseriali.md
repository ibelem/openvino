# Security finding #168: At line 87, `part_size` is assigned directly from the deserialized …

**Summary:** At line 87, `part_size` is assigned directly from the deserialized …

**CWE IDs:** CWE-369: Divide By Zero (also CWE-1284: Improper Validation of Specified Quantity in Input)
**Severity / Impact:** An attacker-supplied model (ONNX or OpenVINO IR) with a TensorIterator whose SliceInputDescription carries `m_part_size=0` causes an integer divide-by-zero during shape inference (validate_and_infer_types), which is triggered at model-load time — before any runtime execution. On Linux/Unix this raises SIGFPE and crashes the hosting process. On Windows it raises an exception. Any application that loads untrusted models (model serving infrastructure, conversion tools, ORT OpenVINO EP) is affected.
**Affected location:** `targets/openvino/src/core/src/op/tensor_iterator.cpp:98` — `op::v0::TensorIterator::validate_and_infer_types()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted model bytes (ONNX/IR) → ONNX/IR deserializer → SliceInputDescription::m_part_size field → validate_and_infer_types

## Description / Root cause
At line 87, `part_size` is assigned directly from the deserialized field `slice_input_description->m_part_size` (declared as `int64_t m_part_size{0}` in multi_subgraph_base.hpp:99 — default 0). At line 98, the code performs integer division `(std::abs(end - start) + 1) / part_size` with no preceding check that `part_size != 0`. No NODE_VALIDATION_CHECK or any other guard exists between lines 87 and 98 to enforce `part_size > 0`.

**Validator analysis:** The flaw is real: tensor_iterator.cpp line 98 computes (std::abs(end - start) + 1) / part_size where part_size = slice_input_description->m_part_size (line 87), and m_part_size defaults to 0 (multi_subgraph_base.hpp:99) with no NODE_VALIDATION_CHECK between assignment (87) and division (98). The division is gated by input_partial_shape.rank().is_static() (86) and input_partial_shape[axis].is_static() (92), both satisfiable with a static-shaped input, so part_size==0 with an integer numerator triggers integer divide-by-zero (SIGFPE on Linux). CWE-369 is accurate; CWE-1284 is a reasonable secondary. Impact is accurate: it fires during validate_and_infer_types at model-load/read time before execution. The same unchecked pattern recurs at line 179 (try_to_set_num_iterations_if_no_slice_inputs: concat->m_part_size) and the output ConcatOutputDescription multiply at 148-149 is benign (multiply, not divide) but its part_size==0 would silently collapse the dim. The proposed fix (NODE_VALIDATION_CHECK(this, part_size > 0, ...) after line 87) is correct and sufficient for the slice-input path; it should ALSO be applied to the concat path at line 179 to fully close the divide-by-zero (the validator correctly notes the output path should also be guarded). I mark openvinoEp rejected rather than na because the defect propagates downstream of OV in principle, but reachability from the EP's ONNX trust boundary specifically requiring m_part_size=0 was not provable from the cited code, whereas the IR-deserializer/public-API path into OV core is clearly proven.

## Exploit / Proof of Concept
Craft an ONNX model containing a TensorIterator node with `input_descriptions` that include a SliceInputDescription with `m_part_size=0`. When the model is loaded and compiled (`Core::compile_model` or `Core::read_model`), OpenVINO calls `validate_and_infer_types()`, assigns `part_size = 0` at line 87, and then divides by zero at line 98 (`(std::abs(end - start) + 1) / 0`), immediately crashing the process.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/core/src/op/tensor_iterator.cpp:98
//     m_num_iterations = (std::abs(end - start) + 1) / part_size;
// where part_size == slice_input_description->m_part_size (line 87), default 0
// (multi_subgraph_base.hpp:99) and no guard exists between 87 and 98.
//
// Pre-fix: with a static-shaped sliced input and part_size==0 this performs an
//   integer divide-by-zero during validate_and_infer_types() -> SIGFPE (ASan/UBSan
//   reports "division by zero").
// Post-fix: the added NODE_VALIDATION_CHECK(this, part_size > 0, ...) converts it
//   into an ov::Exception (ov::NodeValidationFailure), which this test asserts.
//
// Mirrors the existing builder style in
//   openvino/src/core/tests/type_prop/tensor_iterator.cpp.

#include "openvino/op/tensor_iterator.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, tensor_iterator_slice_input_zero_part_size_throws) {
    // Static-shaped external input so both rank.is_static() (line 86) and
    // input_partial_shape[axis].is_static() (line 92) are satisfied, forcing
    // execution of the division at line 98.
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    auto Xi = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = make_shared<op::v1::Multiply>(Xi, M_body);
    auto body = make_shared<ov::Model>(OutputVector{Zo}, ParameterVector{Xi, M_body});

    auto ti = make_shared<op::v0::TensorIterator>();
    ti->set_body(body);
    ti->set_invariant_input(M_body, M);

    // start=0, stride=2, part_size=0 (malicious), end=39, axis=1.
    // set_sliced_input() internally calls validate_and_infer_types(), which
    // reaches tensor_iterator.cpp:98 and divides by zero pre-fix.
    EXPECT_THROW(ti->set_sliced_input(Xi, X, /*start*/ 0, /*stride*/ 2,
                                      /*part_size*/ 0, /*end*/ 39, /*axis*/ 1),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests (cmake --build . --target ov_core_unit_tests). Run: ./ov_core_unit_tests --gtest_filter='type_prop.tensor_iterator_slice_input_zero_part_size_throws'. Pre-fix expectation: process aborts with SIGFPE / UBSan 'division by zero' in op::v0::TensorIterator::validate_and_infer_types at tensor_iterator.cpp:98 (test fails). Post-fix expectation: NODE_VALIDATION_CHECK throws ov::NodeValidationFailure (an ov::Exception) and the test passes.

## Suggested fix
Add a NODE_VALIDATION_CHECK immediately after line 87 (or guard the whole division block): `NODE_VALIDATION_CHECK(this, part_size > 0, "SliceInputDescription m_part_size must be greater than zero, got: ", part_size);`. Alternatively, wrap the division: `if (part_size > 0) { m_num_iterations = (std::abs(end - start) + 1) / part_size; } else { NODE_VALIDATION_CHECK(this, false, "m_part_size must be positive"); }`. The same guard should be applied to ConcatOutputDescription::m_part_size used in the output loop below.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #168.
