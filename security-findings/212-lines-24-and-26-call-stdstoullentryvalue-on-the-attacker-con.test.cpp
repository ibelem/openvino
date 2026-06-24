// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (std::stoull on attacker-controlled
// external_data offset/length). Pre-fix: a non-numeric/overflowing offset value escapes as
// std::invalid_argument / std::out_of_range (NOT an ov::Exception), so EXPECT_THROW(...,
// ov::Exception) fails (uncaught logic_error). Post-fix: the constructor converts it to
// error::invalid_external_data (an ov::Exception) and the assertion passes.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: a compilable test requires a crafted .onnx model whose tensor uses
//       data_location=EXTERNAL with external_data {key:"offset", value:"not_a_number"}.
//       That binary fixture is not authored here, hence this is a SKELETON.
#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers used by onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

// TODO: create models/external_data/bad_offset_not_a_number.onnx — a model with one
//       initializer tensor whose data_location=EXTERNAL and external_data contains
//       {key:"offset", value:"not_a_number"} (or value:"99999999999999999999999999").
TEST(onnx_external_data, malformed_offset_throws_ov_exception) {
    // Pre-fix this aborts/escapes via std::invalid_argument; post-fix it is invalid_external_data.
    EXPECT_THROW(convert_model("external_data/bad_offset_not_a_number.onnx"), ov::Exception);
}

// TODO: create models/external_data/bad_length_overflow.onnx with external_data
//       {key:"length", value:"99999999999999999999999999"} (> ULLONG_MAX).
TEST(onnx_external_data, overflow_length_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_length_overflow.onnx"), ov::Exception);
}
