# Security finding #172: At line 98, `part_size` is taken directly from `slice_input_descrip…

**Summary:** At line 98, `part_size` is taken directly from `slice_input_descrip…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Same as finding 1: SIGFPE / unhandled crash of the host process during model load / shape inference when a crafted model supplies a SliceInputDescription with `m_part_size = 0`.
**Affected location:** `targets/openvino/src/core/src/op/tensor_iterator.cpp:98` — `op::v0::TensorIterator::validate_and_infer_types()`
**Validated for repos:** openvino
**Trust boundary:** Deserialized ONNX/IR model → ov::op::v0::TensorIterator::validate_and_infer_types

## Description / Root cause
At line 98, `part_size` is taken directly from `slice_input_description->m_part_size` (also `int64_t m_part_size{0}`, multi_subgraph_base.hpp:99) and used as the divisor in `(std::abs(end - start) + 1) / part_size` with no preceding zero check. The code only guards with `input_partial_shape.rank().is_static()` and `input_partial_shape[axis].is_static()`, neither of which prevents a zero `m_part_size` from reaching the division.

**Validator analysis:** The vuln type (CWE-369 Divide By Zero) is accurate: at tensor_iterator.cpp:98 `m_num_iterations = (std::abs(end - start) + 1) / part_size` performs int64_t integer division where the divisor is `slice_input_description->m_part_size` (default 0, multi_subgraph_base.hpp:99). The surrounding guards (input_partial_shape.rank().is_static() at L86, input_partial_shape[axis].is_static() at L92) constrain only the shape rank/dimension, never the user-controlled part_size, so a zero value flows straight to the division → SIGFPE / unhandled host crash during shape inference at model load. The stated impact is accurate for the IR/model-load path. The proposed fix is correct and sufficient: a `NODE_VALIDATION_CHECK(this, part_size != 0, ...)` immediately before L98 converts the crash into a catchable ov::NodeValidationFailure. Strengthen it to reject part_size < 0 as well (`part_size > 0`), since a negative part_size also yields meaningless iteration counts and is set into out_shape[axis] at L90. The optional deserialization-boundary check is defense-in-depth but not required once the core op validates. For the EP repo the finding does not propagate: the ONNX frontend does not construct v0 TensorIterator, so the defect cannot be triggered from the EP's ONNX trust boundary.

## Exploit / Proof of Concept
Craft an ONNX TensorIterator node with a SliceInputDescription whose `part_size = 0`. When `validate_and_infer_types` processes input descriptions (lines 82–102), it enters the static-rank branch (line 86), sets `part_size = 0` (line 87), then divides at line 98, triggering integer division by zero.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 in op::v0::TensorIterator::validate_and_infer_types.
// Encodes the fix for:
//   targets/openvino/src/core/src/op/tensor_iterator.cpp:98
//     m_num_iterations = (std::abs(end - start) + 1) / part_size;
// where part_size == slice_input_description->m_part_size (default 0,
//   targets/openvino/src/core/include/openvino/op/util/multi_subgraph_base.hpp:99).
//
// Pre-fix: a SliceInputDescription with part_size == 0 reaches the division at
//   line 98 and triggers an integer divide-by-zero (SIGFPE / FPE_INTDIV abort).
// Post-fix: a NODE_VALIDATION_CHECK rejects part_size == 0, raising
//   ov::NodeValidationFailure (subclass of ov::Exception) instead of crashing.
//
// Style mirrors src/core/tests/type_prop/tensor_iterator.cpp.

#include "openvino/op/tensor_iterator.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, tensor_iterator_slice_input_zero_part_size_throws) {
    // Outer sliced input.
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{32, 40, 10});

    // Trivial body: Xi -> Add(Xi, Xi) -> Zo
    auto Xi = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = make_shared<op::v1::Add>(Xi, Xi);
    auto body = make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi});

    auto ti = make_shared<op::v0::TensorIterator>();
    ti->set_body(body);

    // set_sliced_input(body_parameter, value, start, stride, part_size, end, axis)
    // part_size == 0 is the crafted malicious value (the divisor at line 98).
    ti->set_sliced_input(Xi, X, /*start=*/0, /*stride=*/0, /*part_size=*/0, /*end=*/-1, /*axis=*/1);
    ti->get_iter_value(Zo, -1);

    // Pre-fix this aborts with SIGFPE inside validate_and_infer_types (line 98).
    // Post-fix the NODE_VALIDATION_CHECK turns it into a catchable exception.
    EXPECT_THROW(ti->validate_and_infer_types(), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests (cmake --build . --target ov_core_unit_tests). Run: ./ov_core_unit_tests --gtest_filter='type_prop.tensor_iterator_slice_input_zero_part_size_throws'. Pre-fix expectation: process aborts with SIGFPE / 'Floating point exception (integer divide by zero)' at tensor_iterator.cpp:98 (UBSan: 'division by zero'); test fails to complete. Post-fix expectation: ov::NodeValidationFailure is thrown and the EXPECT_THROW passes.

## Suggested fix
Add a validation check before line 98:
```cpp
NODE_VALIDATION_CHECK(this, part_size != 0,
    "SliceInputDescription m_part_size must not be zero");
m_num_iterations = (std::abs(end - start) + 1) / part_size;
```
The same check should be applied at the deserialization boundary (ONNX importer for TensorIterator) to reject `part_size <= 0` before the IR is constructed.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #172.
