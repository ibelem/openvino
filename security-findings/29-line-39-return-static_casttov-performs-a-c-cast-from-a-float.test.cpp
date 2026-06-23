// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for work item 29 (CWE-20 float->int UB on NaN/Inf).
// Unchecked code: targets/openvino/src/core/reference/include/openvino/reference/convert.hpp:38-39
//   `return static_cast<TO>(v);`  -- no guard against NaN/+-Inf before the
//   float-to-integer cast. Reached via op::v0::Convert::evaluate ->
//   convert::Evaluate::visit -> EvalByOutputType::visit ->
//   reference::convert (iterator overload) -> detail::convert.
//
// Pre-fix: under UBSan this triggers `runtime error: ... is outside the range
//   of representable values of type 'int'` (float-cast-overflow), or yields an
//   indeterminate integer. Post-fix (NaN/Inf saturated to 0 / min / max) the
//   evaluate succeeds and produces finite, deterministic integers.
//
// Goes in src/core/tests/eval.cpp style (target: ov_core_unit_tests), inside
// namespace ov::test. Uses the same make_tensor / model->evaluate helpers.

TEST(eval, evaluate_convert_f32_nan_inf_to_i32_no_ub) {
    using ov::op::v0::Parameter;
    using ov::op::v0::Convert;

    auto p = make_shared<Parameter>(element::f32, PartialShape{4});
    auto cvt = make_shared<Convert>(p, element::i32);
    auto model = make_shared<Model>(OutputVector{cvt}, ParameterVector{p});

    // NaN (0x7FC00000), +Inf (0x7F800000), -Inf (0xFF800000), and a normal value.
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();

    auto result = ov::Tensor();
    auto out_vector = ov::TensorVector{result};
    auto in_vector =
        ov::TensorVector{make_tensor<element::Type_t::f32>(Shape{4}, {nan_v, pinf, ninf, 7.0f})};

    // Pre-fix this evaluate executes the unguarded cast (UBSan abort on the
    // float-cast-overflow). Post-fix it must complete and saturate.
    ASSERT_TRUE(model->evaluate(out_vector, in_vector));
    result = out_vector.at(0);
    EXPECT_EQ(result.get_element_type(), element::i32);
    EXPECT_EQ(result.get_shape(), (Shape{4}));

    auto vals = read_vector<int32_t>(result);
    // The finite input must convert exactly; the special values must be finite
    // (exact saturated values depend on the chosen fix: NaN->0, +Inf->INT32_MAX,
    // -Inf->INT32_MIN). The key invariant is: no UB, deterministic finite output.
    EXPECT_EQ(vals[3], 7);
    EXPECT_EQ(vals[0], 0);                                  // NaN -> 0 (per proposed fix)
    EXPECT_EQ(vals[1], std::numeric_limits<int32_t>::max());  // +Inf saturates
    EXPECT_EQ(vals[2], std::numeric_limits<int32_t>::min());  // -Inf saturates
}