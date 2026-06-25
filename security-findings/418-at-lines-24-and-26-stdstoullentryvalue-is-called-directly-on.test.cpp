// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor_external_data.cpp:24,26 (std::stoull on unvalidated
// attacker-controlled external_data 'offset'/'length' strings).
//
// Pre-fix: convert_model on a model whose initializer has data_location=EXTERNAL
// and external_data offset="AAAA" (or length="99999999999999999999999") lets
// std::stoull throw std::invalid_argument / std::out_of_range, which is a
// std::logic_error, NOT an ov::Exception -> this EXPECT_THROW(..., ov::Exception)
// FAILS (wrong exception type escapes the frontend).
// Post-fix: the constructor wraps std::stoull and rethrows
// error::invalid_external_data (an ov::Exception) -> assertion passes.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
// NOTE: needs a crafted .onnx fixture (see TODO) -> skeleton.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_bad_offset_string_throws_ov_exception) {
    // TODO: add the crafted model fixture under
    //   onnx/frontend/tests/models/external_data/external_data_bad_offset.onnx
    // containing one initializer with data_location=EXTERNAL and an
    // external_data entry {key="offset", value="AAAA"} (non-numeric).
    // Build the model object the same way other onnx_import.in.cpp tests do, e.g.:
    //   const auto model = convert_model("external_data/external_data_bad_offset.onnx");
    EXPECT_THROW(convert_model("external_data/external_data_bad_offset.onnx"),
                 ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_length_throws_ov_exception) {
    // TODO: fixture external_data_overflow_length.onnx with external_data entry
    //   {key="length", value="99999999999999999999999"} (> ULLONG_MAX) to drive
    //   the std::out_of_range branch at tensor_external_data.cpp:26.
    EXPECT_THROW(convert_model("external_data/external_data_overflow_length.onnx"),
                 ov::Exception);
}