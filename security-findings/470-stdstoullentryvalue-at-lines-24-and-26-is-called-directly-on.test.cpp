// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for CWE-248 at tensor_external_data.cpp:24/26
// (std::stoull on attacker-controlled external_data offset/length strings).
// Pre-fix: convert_model throws std::invalid_argument (uncaught by ov::Exception handlers) -> abort/DoS.
// Post-fix: ctor converts to error::invalid_external_data (an ov::Exception subclass), so EXPECT_THROW(ov::Exception) passes cleanly.

#include "onnx_utils.hpp"  // FrontEndTestUtils / convert_model helper used by onnx_import.in.cpp
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted model fixture, e.g. models/external_data/invalid_offset_nonnumeric.onnx,
//       containing a TensorProto with external_data: [{key:"location",value:"data.bin"},
//       {key:"offset",value:"not_a_number"}]. Without the fixture this test cannot run.
TEST(onnx_external_data, invalid_external_data_offset_string_is_rejected) {
    // Pre-fix this surfaces as an uncaught std::invalid_argument from std::stoull (DoS).
    // Post-fix the ctor must translate it into ov::Exception (error::invalid_external_data).
    EXPECT_THROW(convert_model("external_data/invalid_offset_nonnumeric.onnx"), ov::Exception);
}

TEST(onnx_external_data, empty_external_data_length_string_is_rejected) {
    // TODO: fixture with external_data length value == "" (empty) -> std::stoull throws std::invalid_argument.
    EXPECT_THROW(convert_model("external_data/empty_length.onnx"), ov::Exception);
}