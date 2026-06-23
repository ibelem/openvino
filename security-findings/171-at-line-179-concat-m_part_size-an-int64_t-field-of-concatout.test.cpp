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