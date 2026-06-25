// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for tensor_external_data.cpp:24,26 (std::stoull on
// attacker-controlled external_data 'offset'/'length' strings, unguarded).
// Pre-fix: convert_model throws std::invalid_argument (not ov::Exception),
//          which EXPECT_THROW(..., ov::Exception) FAILS to match -> test fails.
// Post-fix: constructor catches and rethrows error::invalid_external_data
//          (an ov::Exception subclass) -> EXPECT_THROW matches -> test passes.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture model whose TensorProto.external_data carries
//       {key:"location", value:"data.bin"} and {key:"offset", value:"not_a_number"}
//       (and a sibling case with length="99999999999999999999999999").
//       These binary .onnx fixtures must be added under the onnx frontend test
//       models dir; see TODO below.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_invalid_offset_string) {
    // TODO: add fixture 'external_data_invalid_offset.onnx' under
    //       onnx/frontend/tests/models/ with external_data offset="not_a_number".
    EXPECT_THROW(convert_model("external_data_invalid_offset.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_length_string) {
    // TODO: add fixture 'external_data_overflow_length.onnx' under
    //       onnx/frontend/tests/models/ with external_data length="99999999999999999999999999".
    EXPECT_THROW(convert_model("external_data_overflow_length.onnx"), ov::Exception);
}