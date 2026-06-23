# Security finding #29: Line 39: `return static_cast<TO>(v);` performs a C++ cast from a fl…

**Summary:** Line 39: `return static_cast<TO>(v);` performs a C++ cast from a fl…

**CWE IDs:** CWE-20: Improper Input Validation (C++ Undefined Behaviour: float-to-integer cast of NaN/Inf)
**Severity / Impact:** Undefined behaviour during constant folding of a model with NaN/Inf float constants converted to integer types (i8, i16, i32, i64, u8, …). Depending on compiler and optimisation level this can produce silently wrong integer constants that propagate through the rest of the inference graph, or — in adversarial compiler-codegen scenarios — memory corruption. Affects all callers that do not go through the Clamp specialisation (which itself has the same bug for NaN as noted below), including the NoClamp path used for float→int8.
**Affected location:** `targets/openvino/src/core/reference/include/openvino/reference/convert.hpp:38` — `ov::reference::detail::convert<TI, TO>()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX/IR model file — float constants embedded in the model graph reach Convert::evaluate during constant folding

## Description / Root cause
Line 39: `return static_cast<TO>(v);` performs a C++ cast from a floating-point type to an integer type with no guard against IEEE special values. Per C++ [conv.fpint], casting NaN or ±Inf to any integer type is undefined behaviour; the compiler is free to generate any code, including code that corrupts adjacent memory, traps, or produces security-relevant wrong values in the folded constant tensor.

**Validator analysis:** The flaw is real: casting a floating-point NaN or ±Inf to an integer type is C++ undefined behaviour ([conv.fpint]), and convert.hpp:38-39 performs that cast with zero guard. It is reachable from an untrusted model: a f32 Constant holding 0x7FC00000 (NaN) or 0x7F800000 (Inf) feeding a Convert→i8/i16/i32/i64/u* node is constant-folded through Convert::evaluate. Crucially, on this path the arguments are passed to reference::convert as element iterators (core/src/op/convert.cpp:131 wraps both args in element::iterator<>), so the InputIt/OutputIt overload (convert.hpp:48) is selected — NOT the TI*/TO* specialisations and NOT the JIT/convert_impl kernels. Thus even float→int8 (and entirely float→i16/i32/i64/u* which have no JIT kernel at all) take the scalar `detail::convert` cast at line 39. The CWE-20 / float-cast UB classification is accurate. However the stated impact is OVERSTATED: the output tensor is sized exactly to `count` (out.set_shape(in_shape) at convert.cpp:212) and exactly `count` elements are written, so there is no OOB write or adjacent-memory corruption — the realistic consequence is a UBSan `float-cast-overflow` trap (or a deterministically wrong-but-bounded folded integer constant) rather than memory corruption. The Clamp path (convert_util.hpp:38-43) has the same NaN hole: NaN fails both `<lowest()` and `>max()` comparisons and still reaches the raw cast. The proposed fix is correct in direction and sufficient to remove the UB, but it should be applied to the integral-output branch of detail::convert AND mirrored in Clamp::apply (convert_util.hpp:38) since NaN bypasses both clamp comparisons; clamping ±Inf to numeric_limits min/max and mapping NaN to 0 matches IEEE-754 saturating-conversion semantics and is preferable to the silent value change the SFINAE-only fix introduces. Note the fix must guard the iterator-overload scalar path that is actually taken here, not just the TI* specialisations.

## Exploit / Proof of Concept
An attacker supplies a model (ONNX or OpenVINO IR) that contains a float32 Constant tensor whose payload bytes encode NaN (e.g. 0x7FC00000) or +Inf (0x7F800000), followed by a Convert node targeting i8/i16/i32/i64/u8. During constant folding, `Convert::evaluate` (convert.cpp:202-219) dispatches through `Evaluate::visit` → `EvalByOutputType::visit` (convert.cpp:130-133) → `reference::convert<float,int8_t>` (convert.hpp:63-65) → `detail::convert<float,int8_t>` (convert.hpp:38-39) with `v = NaN`. No check precedes the cast. On the non-JIT scalar path this is C++ UB.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
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
```
**Build / run:** Build target: ov_core_unit_tests. Run: ov_core_unit_tests --gtest_filter='eval.evaluate_convert_f32_nan_inf_to_i32_no_ub'. With -fsanitize=undefined the PRE-FIX run aborts at convert.hpp:39 with UBSan 'runtime error: ... is outside the range of representable values of type int' (float-cast-overflow); after the NaN/Inf guard is added the test passes. If the fix maps NaN/Inf to different sentinels than assumed, adjust the EXPECT_EQ on vals[0..2] (the no-UB + finite invariant is the core regression).

## Suggested fix
Add a NaN/Inf guard in `detail::convert` when `TO` is an integral type:
```cpp
template <typename TI, typename TO>
constexpr typename std::enable_if<!std::is_same<TO,char>::value &&
                                   std::is_integral<TO>::value &&
                                   std::is_floating_point<TI>::value, TO>::type
convert(const TI v) {
    if (!std::isfinite(static_cast<double>(v)))
        return static_cast<TO>(0);  // or clamp to min/max
    return static_cast<TO>(v);
}
```
Alternatively, always dispatch float→integer through `Clamp<TI,TO>` (which then also needs the NaN fix described below).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #29.
