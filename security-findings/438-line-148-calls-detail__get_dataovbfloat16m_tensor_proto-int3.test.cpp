// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for tensor.cpp:148 (and the twin defect at line 142):
//   non-raw BFLOAT16 initializers stored as 16-bit bit patterns in int32_data
//   must decode via ov::bfloat16::from_bits, not numeric int->float conversion.
// Pre-fix: the constant value reads ~16256.0 (0x3F80 reinterpreted numerically)
//   instead of 1.0 -> assertion fails.
// Post-fix: from_bits decode yields 1.0 -> assertion passes.
// Harness: ov_onnx_frontend_tests, TEST in the style of onnx_import.in.cpp.
//
// NOTE (skeleton): this needs a crafted .prototxt fixture under the onnx test
// models dir declaring a BFLOAT16 Constant/initializer whose int32_data element
// = 16256 (0x3F80 == bfloat16 1.0). Symbol/path TODOs flagged below.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm helper header name in this test dir
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace

OPENVINO_TEST(${FRONTEND_NAME}_onnx, model_bfloat16_initializer_nonraw_from_bits) {
    // TODO: create models/bfloat16_const_nonraw.prototxt:
    //   a graph with one BFLOAT16 initializer/Constant, dims=[1],
    //   int32_data: 16256   (0x3F80 == bit pattern of bfloat16 1.0)
    const auto model = convert_model("bfloat16_const_nonraw.prototxt");

    std::shared_ptr<op::v0::Constant> cst;
    for (const auto& node : model->get_ordered_ops()) {
        if ((cst = ov::as_type_ptr<op::v0::Constant>(node))) break;
    }
    ASSERT_NE(cst, nullptr);
    ASSERT_EQ(cst->get_element_type(), element::bf16);

    const auto values = cst->cast_vector<float>();
    ASSERT_EQ(values.size(), 1u);
    // Pre-fix this is ~16256.0f; the fix must decode the bit pattern to 1.0f.
    EXPECT_FLOAT_EQ(values[0], 1.0f);
}