// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor_external_data.cpp:24,26 (TensorExternalData ctor).
// Pre-fix: std::stoull on a malformed offset/length string ('' or 'abc' or a
//   26-digit overflow) throws std::invalid_argument / std::out_of_range, a raw
//   std exception that is NOT an ov::Exception -> EXPECT_THROW(..., ov::Exception)
//   FAILS pre-fix (wrong exception type escapes).
// Post-fix: ctor wraps stoull and rethrows error::invalid_external_data (an
//   ov::Exception subclass) -> assertion passes.
//
// Style mirrors onnx_import.in.cpp; uses convert_model() helper.
// SKELETON: requires a crafted ONNX fixture with an initializer whose
//   data_location=EXTERNAL and external_data entry key='offset' value=''.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: create fixture models/external_data/bad_offset_empty.onnx with an
//       initializer: data_location=EXTERNAL, external_data{key:'offset',value:''}
//       (and similarly bad_length_nan.onnx with key:'length',value:'abc',
//        and bad_offset_overflow.onnx with value:'99999999999999999999999999').
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_offset_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_offset_empty.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_length_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_length_nan.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_offset_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_offset_overflow.onnx"), ov::Exception);
}