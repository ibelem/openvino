# Security finding #427: At line 139 the guard `product_out.is_static()` only checks that `o…

**Summary:** At line 139 the guard `product_out.is_static()` only checks that `o…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Any caller that loads a crafted model containing a Reshape node with a zero static output dimension and a -1 dimension (e.g. pattern [-1, 0, N] with special_zero=false) will trigger undefined behavior during shape inference. In practice this causes a crash (DoS) because Inf/NaN cast to int64_t has implementation-defined or trap behavior on most platforms. In optimized builds the UB may corrupt internal Dimension state before the validation fires, opening a path to further exploitation. The affected surface is the OpenVINO shape inference engine, reachable during model loading/compilation.
**Affected location:** `targets/openvino/src/core/shape_inference/include/reshape_shape_inference.hpp:139` — `reshape::resolve_minus_one_dim<TDim> (Dimension overload)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted model reshape pattern loaded by the OpenVINO runtime; attacker controls the reshape pattern values fed into shape inference

## Description / Root cause
At line 139 the guard `product_out.is_static()` only checks that `out.second` (the dynamic accumulator from `Product<Dimension>`) is a known/static Dimension — it does NOT check that the value is non-zero. When `out.second` is a static Dimension with value 0, `product_out.get_length()` (dimension.cpp:206–211) returns 0 without throwing. The subsequent `minus_one_dim /= product_out.get_length()` at line 140 calls `Dimension::operator/(value_type)` (dimension.cpp:125–134) with divisor=0. The guard in that operator — `OPENVINO_ASSERT(divisor >= 0)` (dimension.cpp:126) — is a logic bug: zero satisfies `>= 0`, so the assertion PASSES and does not block the zero divisor. The code then executes `ceil(double_min_val / 0.0)` and `floor(double_max_val / 0.0)`, producing ±infinity or NaN, followed by a cast to `int64_t`, which is undefined behavior in C++. The post-hoc validation at line 349 (`if (product.get_static_out() == 0)`) fires only after `resolve_minus_one_dim` has already returned, so it cannot prevent the UB.

**Validator analysis:** Confirmed by tracing the cited path. For pattern [-1,0,3], special_zero=false, input [6]: product.calculate() (lines 90-117) yields in.second=Dimension(6) and out.second=Dimension(0); resolve_minus_one_dim sees both static (line 139) and calls minus_one_dim/=0 (line 140). Dimension::operator/ (dimension.cpp:125-134) asserts divisor>=0 — a genuine logic bug since the message says 'greater than 0' but 0 satisfies >=0 — and the early-return guard at line 129 only fires when max==s_max && min==0, which is false for the static 6, so execution reaches static_cast<int64_t>(ceil(6.0/0.0)) i.e. cast of +inf to int64_t, which is C++ UB. The validation at lines 349-353 fires only after resolve_minus_one_dim returns, so it cannot prevent the UB. vulnType CWE-369 is slightly imprecise (the integer divide-by-zero is actually a double 6.0/0.0→inf, with the real UB being the inf→int64_t cast), but the defect and DoS impact are real. proposedFix is correct and sufficient: fix #2 (assert divisor>0) converts the silent passthrough into a thrown ov::Exception, and fix #1 (early return TDim{0} when product_out==0, deferring rejection to the line-349 check) gives a clean fallback; the same divisor==0 risk also exists in the static overload at line 167, which the operator fix covers. Both repos are on the reachable path; the fix lands in openvino core.

## Exploit / Proof of Concept
Provide a Reshape node with: input shape [6] (static), reshape pattern tensor value [-1, 0, 3], and special_zero=false. The `0` is not a special-zero, so `product.update_out(Dimension(0))` is called at line 325. After `product.calculate()` (line 341): `mul(out, Dimension(0))` sets `out.first = Dimension(0)*Dimension(1) = Dimension(0)`; since `in.first (=Dimension(6)) != out.first (=Dimension(0))`, line 111–112 execute and `out.second *= out.first` → `Dimension(1) * Dimension(0) = Dimension(0)`. In `resolve_minus_one_dim`, `product.get_dynamic_out()` returns `Dimension(0)` which is static (Interval [0,0], size=1). `product_out.get_length()` returns 0. `Dimension::operator/(0)` is invoked; `OPENVINO_ASSERT(0 >= 0)` passes. Then `static_cast<double>(min_val) / 0` = ±Inf or NaN; `static_cast<int64_t>(Inf)` is UB. The check at line 349 never gets a chance to reject the input.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression for CWE-369/UB at reshape_shape_inference.hpp:139-140 and dimension.cpp:126.
// Pre-fix: shape inference of Reshape(input [6], pattern [-1,0,3], special_zero=false)
// reaches Dimension::operator/(0) (assert divisor>=0 passes) -> static_cast<int64_t>(ceil(6.0/0.0))
// which is undefined behavior (UBSan: 'inf is outside the range of int64_t').
// Post-fix: input is rejected with ov::Exception (NODE_VALIDATION_CHECK at line 349-353)
// or the divisor assertion fires, instead of UB.
//
// TODO: confirm exact target/harness — likely ov_core_unit_tests; place near existing
//       reshape shape-inference tests (e.g. src/core/tests/type_prop/reshape.cpp or
//       src/plugins/.../shape_inference tests). Verify includes/symbol names below.

#include <gtest/gtest.h>

#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(type_prop_reshape, minus_one_with_zero_static_output_dim_no_ub) {
    // input shape [6], reshape pattern [-1, 0, 3], special_zero = false
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6});
    auto pattern = op::v0::Constant::create(element::i64, Shape{3}, {-1, 0, 3});
    // special_zero = false -> the 0 is a real (zero-size) output dim, not 'copy from input'
    // Pre-fix this triggers UB in Dimension::operator/ during shape inference.
    // Post-fix it must throw a controlled ov::Exception (cannot infer '-1' with zero-size out).
    EXPECT_THROW(
        { auto r = std::make_shared<op::v1::Reshape>(data, pattern, /*special_zero=*/false); (void)r; },
        ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests (verify the actual target that owns type_prop/reshape tests). Run: ./ov_core_unit_tests --gtest_filter='type_prop_reshape.minus_one_with_zero_static_output_dim_no_ub'. Pre-fix with -fsanitize=undefined expect: 'runtime error: inf is outside the range of representable values of type int64_t' (dimension.cpp:131-132); post-fix expect the EXPECT_THROW(ov::Exception) to be satisfied.

## Suggested fix
1) In `resolve_minus_one_dim` (reshape_shape_inference.hpp:139–140), add an explicit zero check before dividing: `if (minus_one_dim.is_static() && product_out.is_static()) { if (product_out.get_length() == 0) return TDim{0}; // caught by post-hoc check at line 349 minus_one_dim /= product_out.get_length(); }`. 2) Fix the logic bug in `Dimension::operator/` (dimension.cpp:126): change `OPENVINO_ASSERT(divisor >= 0, "divisor must be greater than 0")` to `OPENVINO_ASSERT(divisor > 0, "divisor must be greater than 0")` so that a zero divisor actually triggers the assertion instead of silently passing through. Both fixes are needed: the operator fix closes the general case, and the early return in `resolve_minus_one_dim` provides a safe fallback that defers rejection to the existing validation at line 349.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #427.
