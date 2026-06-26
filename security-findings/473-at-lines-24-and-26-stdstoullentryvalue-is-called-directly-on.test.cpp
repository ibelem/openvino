// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (TensorExternalData ctor std::stoull on
// untrusted external_data 'offset'/'length'). Pre-fix: convert_model on a model whose external_data
// 'offset' (or 'length') value is non-numeric throws std::invalid_argument (uncaught logic_error),
// failing EXPECT_THROW(..., ov::Exception). Post-fix it is converted to error::invalid_external_data
// (an ov::Exception), so the assertion passes.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture model (see TODO) because TensorExternalData is constructed from a
// TensorProto deep inside Tensor::get_external_data (tensor.hpp:318-322) reached only via convert_model.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: add models/external_data/invalid_offset_nonnumeric.onnx + .data where a TensorProto has
//       data_location=EXTERNAL and external_data entry key="offset", value="abc" (and a sibling
//       fixture with key="length", value="99999999999999999999999999999" for the out_of_range case).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_invalid_offset_string) {
    EXPECT_THROW(convert_model("external_data/invalid_offset_nonnumeric.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_length_out_of_range) {
    EXPECT_THROW(convert_model("external_data/invalid_length_overflow.onnx"), ov::Exception);
}
