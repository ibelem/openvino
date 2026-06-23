# Security finding #171: At line 179, `concat->m_part_size` (an `int64_t` field of `ConcatOu…

**Summary:** At line 179, `concat->m_part_size` (an `int64_t` field of `ConcatOu…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Signed integer division by zero is undefined behavior in C++; on all common platforms it produces SIGFPE/structured exception, causing an unhandled crash of the inference engine process. Any model consumer (ONNX Runtime with OpenVINO EP, direct OV Core users) that loads a crafted model file will crash during shape inference / model compilation. This is a denial-of-service against the host process; if the model loader runs in a shared service, all tenants are affected.
**Affected location:** `targets/openvino/src/core/src/op/tensor_iterator.cpp:179` — `op::v0::TensorIterator::try_to_set_num_iterations_if_no_slice_inputs()`
**Validated for repos:** openvino
**Trust boundary:** Deserialized ONNX/IR model → ov::op::v0::TensorIterator::try_to_set_num_iterations_if_no_slice_inputs

## Description / Root cause
At line 179, `concat->m_part_size` (an `int64_t` field of `ConcatOutputDescription`, default-initialized to 0 per the header at multi_subgraph_base.hpp:155) is used as the divisor in `(std::abs(concat->m_end - concat->m_start)) / concat->m_part_size` with no preceding zero check. The function enters this path only when `m_num_iterations == -1` and no SliceInputDescription is present (line 173 guard), meaning `m_num_iterations` has never been set from a slice-input path, so there is no upstream guard. The field is populated verbatim from the deserialized model via `visit_attributes` → `AttributeVisitor::on_attribute("output_descriptions", ...)` with no validation of `m_part_size > 0`.

**Validator analysis:** The CWE-369 classification is accurate: int64_t signed division by zero at line 179 is UB and on common platforms raises SIGFPE, crashing the loader process — a DoS, as the impact states. The guard at line 173 only short-circuits when m_num_iterations is already set or a SliceInputDescription exists; with neither, line 179 executes unconditionally with no validation that m_part_size != 0, and the field is set verbatim from visit_attributes (line 18) during IR deserialization. Reachability is genuine for the openvino IR boundary but NOT for the ONNX EP, because the ONNX frontend lowers loops to v5::Loop, so the EP cannot construct a v0::TensorIterator carrying a crafted part_size; hence openvinoEp is rejected on reachability, not on validity. The proposed fix (OPENVINO_ASSERT(concat->m_part_size != 0, ...) before the division) is correct and sufficient to convert the crash into a clean ov::Exception; even better is to also reject part_size <= 0 (negative part_size yields a meaningless/negative iteration count) and to validate part_size in the ConcatOutputDescription constructor / IR importer so the bad value is caught at deserialization rather than at shape inference. Note std::abs(m_end - m_start) does not guard against m_part_size being larger than the range (result 0 iterations) but that is a correctness, not a safety, concern.

## Exploit / Proof of Concept
Craft an ONNX TensorIterator node that has zero SliceInputDescriptions and one ConcatOutputDescription with `part_size = 0` (e.g., `stride=1, start=0, end=10, part_size=0, axis=0`). When `TensorIterator::validate_and_infer_types()` is called (line 123 calls `try_to_set_num_iterations_if_no_slice_inputs()`), the guard at line 173 passes (no slice inputs, `m_num_iterations == -1`), the loop at line 177 finds the `ConcatOutputDescription`, and line 179 performs integer division by zero, crashing the process.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero at
//   openvino/src/core/src/op/tensor_iterator.cpp:179
//   m_num_iterations = (std::abs(concat->m_end - concat->m_start)) / concat->m_part_size;
// ConcatOutputDescription::m_part_size defaults to 0 (multi_subgraph_base.hpp:155).
// When a TensorIterator has NO SliceInputDescription and m_num_iterations == -1,
// the guard at line 173 falls through, the loop finds the ConcatOutputDescription,
// and line 179 divides by m_part_size == 0.
// Pre-fix: integer divide-by-zero -> SIGFPE / UBSan 'division by zero'.
// Post-fix: OPENVINO_ASSERT(part_size != 0, ...) throws ov::Exception.
// Place alongside the existing tests in src/core/tests/type_prop/tensor_iterator.cpp.
TEST(type_prop, tensor_iterator_concat_zero_part_size_no_slice_inputs) {
    using namespace ov;
    // Invariant-input values whose shapes match their body parameters,
    // so no unrelated validation error fires before we reach line 179.
    auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto M = std::make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    auto Xi = std::make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    auto Zo = std::make_shared<op::v1::Multiply>(Xi, M_body);
    auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, M_body});

    auto ti = std::make_shared<op::v0::TensorIterator>();
    ti->set_body(body);
    // Deliberately NO set_sliced_input(): has_slice_input_desc() == false and
    // m_num_iterations stays at its default -1, so the guard at line 173 passes.
    ti->set_invariant_input(Xi, X);
    ti->set_invariant_input(M_body, M);
    // Concatenated output with part_size == 0 (start=0, stride=1, part_size=0, end=10, axis=1).
    ti->get_concatenated_slices(Zo, 0, 1, 0, 10, 1);

    // Pre-fix: divide-by-zero crash at tensor_iterator.cpp:179.
    // Post-fix: clean ov::Exception from the added zero-check.
    EXPECT_THROW(ti->validate_and_infer_types(), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests (cmake --build . --target ov_core_unit_tests). Run: ./ov_core_unit_tests --gtest_filter='type_prop.tensor_iterator_concat_zero_part_size_no_slice_inputs'. Pre-fix expectation: the process aborts before the EXPECT_THROW can succeed — SIGFPE (Floating point exception) on a UBSan build reported as 'tensor_iterator.cpp:179: runtime error: division by zero'. Post-fix expectation: OPENVINO_ASSERT throws ov::Exception and the test passes.

## Suggested fix
Add a zero-check immediately before the division. Replace line 179 with:
```cpp
OPENVINO_ASSERT(concat->m_part_size != 0,
    "ConcatOutputDescription m_part_size must not be zero");
m_num_iterations = (std::abs(concat->m_end - concat->m_start)) / concat->m_part_size;
```
Alternatively, add validation in the `ConcatOutputDescription` constructor and/or in `visit_attributes` / the ONNX importer to reject `part_size <= 0`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #171.
