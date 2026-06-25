# Security finding #424: Line 167 unconditionally computes `product.get_static_in() / produc…

**Summary:** Line 167 unconditionally computes `product.get_static_in() / produc…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Integer divide-by-zero (SIGFPE / undefined behavior) inside shape inference, causing a hard crash (denial of service) of any process that loads a crafted model containing a Reshape node with a 0-valued non-special dimension alongside a -1 dimension. Affects all callers of `v1::shape_infer` on a static shape path, including the OpenVINO EP model loading flow.
**Affected location:** `targets/openvino/src/core/shape_inference/include/reshape_shape_inference.hpp:167` — `reshape::resolve_minus_one_dim (static overload, TDim != Dimension)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled ONNX/OpenVINO model loaded via OpenVINO EP → shape inference path; the reshape pattern tensor is untrusted model data

## Description / Root cause
Line 167 unconditionally computes `product.get_static_in() / product.get_static_out().get_length()`. For the static-dimension specialization, `get_static_out()` returns the raw accumulated `T out` accumulator (line 40-42). When any non-minus-one, non-special-zero pattern dimension equals 0, `Product::update_out()` (line 27-29) multiplies it into `out`, driving `out` to 0. The only guard — `if (product.get_static_out() == 0)` at line 349 — executes *after* `resolve_minus_one_dim` is called at line 347, so the integer divide-by-zero fires first.

**Validator analysis:** Confirmed. The static specialization of Product (lines 18-45) accumulates out via update_out (line 27-29). With pattern [0,-1] and special_zero=false on a fully-static input, the i=0 iteration hits the else branch (line 323-325) calling product.update_out(0), setting out=0; calculate() is a no-op (line 44). With a -1 present, has_minus_one_idx is true, so line 347 calls resolve_minus_one_dim, which at line 167 evaluates get_static_in()/get_static_out().get_length() = N/0 → integer divide-by-zero (SIGFPE). The only zero guard (line 349) executes after. CWE-369 and the DoS/SIGFPE impact are accurate; the static specialization (TDim != Dimension, i.e. StaticDimension) is exactly what plugins use for static shape inference during model compile/load, so it is reachable from the EP. Note the dynamic-Dimension overload (line 139-140) has the same latent issue when product_out is static 0, but the finding correctly targets the static overload which has no guard at all. The proposed fix is correct and sufficient: guarding denom==0 before the division (returning TDim{0}) preserves the subsequent line-349 semantics (it then validates in==0 vs in!=0). Equivalent and arguably cleaner is hoisting the line-349 out==0 check before line 347 and only dividing when out!=0. Either prevents the SIGFPE while keeping the existing validation messages.

## Exploit / Proof of Concept
Craft an ONNX model with a Reshape node whose pattern is `[0, -1]` and `special_zero=False`. The input shape must be fully static (e.g., `[3, 4]`). In `shape_infer`, `special_zero` is false so `ignore_pattern_dim` is false; `product.update_out(0)` is called at line 325, setting `product.out = 0`. `product.calculate()` is a no-op for the static specialisation (line 44). At line 347 `resolve_minus_one_dim(product)` is called, reaching line 167: `product.get_static_in() / product.get_static_out().get_length()` → division by 0 → SIGFPE/crash. The zero-check at line 349 is never reached.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for CWE-369 divide-by-zero in
// openvino/src/core/shape_inference/include/reshape_shape_inference.hpp:167
// (resolve_minus_one_dim static overload). Pre-fix: SIGFPE / FPE on integer
// divide-by-zero when a Reshape pattern contains a 0-valued non-special dim
// alongside a -1 dim on a fully-static input. Post-fix: the op should throw an
// ov::Exception (NODE_VALIDATION_CHECK) rather than crash.
//
// NOTE: This uses the static-shape-inference harness (StaticShape /
// shape_inference()). Confirm exact include paths and helper names against the
// reshape static shape inference test in the tree before building.

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
// TODO: include the static shape inference utility header used by the existing
// reshape static-shape tests (e.g. "utils.hpp" / "common_shape_inference.hpp"
// providing StaticShape and shape_inference(node, in_shapes, out_shapes)).

using namespace ov;

TEST(StaticShapeInferenceTest, ReshapeMinusOneWithZeroDimNoDivByZero) {
    // Reshape pattern [0, -1], special_zero = false, static input [3, 4].
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    auto pattern =
        op::v0::Constant::create(element::i64, ov::Shape{2}, std::vector<int64_t>{0, -1});
    auto reshape = std::make_shared<op::v1::Reshape>(input, pattern, /*special_zero=*/false);

    // TODO: replace with the project's StaticShape vector type and the
    // shape_inference() helper used by reshape_shape_inference tests.
    // std::vector<StaticShape> input_shapes{StaticShape{3, 4}, StaticShape{2}};
    // std::vector<StaticShape> output_shapes;
    // Pre-fix this divides by zero at reshape_shape_inference.hpp:167.
    // Post-fix it must reject via NODE_VALIDATION_CHECK.
    // OV_EXPECT_THROW(shape_inference(reshape.get(), input_shapes, output_shapes),
    //                 ov::Exception, testing::HasSubstr("output dimension"));
    GTEST_SKIP() << "Fill in StaticShape harness symbols from the reshape static "
                    "shape inference test before enabling.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or the static shape-inference test target that hosts reshape_shape_inference_test.cpp). Run: ./ov_cpu_unit_tests --gtest_filter=StaticShapeInferenceTest.ReshapeMinusOneWithZeroDimNoDivByZero . Expected pre-fix: integer divide-by-zero crash (SIGFPE / UBSan 'division by zero' at reshape_shape_inference.hpp:167). Expected post-fix: ov::Exception thrown and test passes.

## Suggested fix
Add a zero-guard inside `resolve_minus_one_dim` (static overload) before the division:
```cpp
template <class TDim, ...>
TDim resolve_minus_one_dim(const Product<TDim>& product) {
    const auto denom = product.get_static_out().get_length();
    if (denom == 0) return TDim{0};  // caller already validates the 0/0 vs nonzero/0 case at line 349
    return product.get_static_in() / denom;
}
```
Alternatively, hoist the `product.get_static_out() == 0` check from line 349 to *before* the `resolve_minus_one_dim` call at line 347, and only call `resolve_minus_one_dim` when `product.out != 0`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #424.
