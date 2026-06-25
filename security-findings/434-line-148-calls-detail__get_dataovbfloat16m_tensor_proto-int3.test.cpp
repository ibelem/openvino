// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:148.
// Pre-fix: __get_data<ov::bfloat16>(int32_data()) numerically converts the stored int32
//          magnitude (e.g. 0x3F80 -> bf16(16256.0)) instead of reinterpreting the low 16
//          bits as a bfloat16 bit-pattern.
// Post-fix: int32_data entries are decoded with ov::bfloat16::from_bits, so 0x3F80 -> 1.0f.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// This needs a crafted .onnx whose BFLOAT16 initializer stores values in int32_data
// (NOT raw_data), so the data lands on tensor.cpp line 147-148. Hence a binary fixture
// is required and the symbol/path details below are best-effort TODOs.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm include used by onnx_import.in.cpp for convert_model()
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace/helper for convert_model

TEST(onnx_import_bfloat16, int32_data_bitpattern_decode) {
    // TODO: provide models/bfloat16_int32_data_const.onnx:
    //   - a single BFLOAT16 Constant/initializer of shape [3]
    //   - data stored in int32_data = {0x3F80, 0xC000, 0x4040}
    //     (bf16 bit-patterns for 1.0, -2.0, 3.0), data_type = BFLOAT16, no raw_data.
    const auto model = convert_model("bfloat16_int32_data_const.onnx");

    std::shared_ptr<ov::op::v0::Constant> cst;
    for (const auto& node : model->get_ordered_ops()) {
        if ((cst = ov::as_type_ptr<ov::op::v0::Constant>(node))) break;
    }
    ASSERT_NE(cst, nullptr);
    ASSERT_EQ(cst->get_element_type(), ov::element::bf16);

    const auto values = cst->cast_vector<float>();
    ASSERT_EQ(values.size(), 3u);
    // Pre-fix these are 16256.0, -49152.0, 49216.0 (numeric ctor); fix yields the bit-patterns.
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], -2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
}
