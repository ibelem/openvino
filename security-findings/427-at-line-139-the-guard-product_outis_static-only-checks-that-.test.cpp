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
