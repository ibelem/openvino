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
