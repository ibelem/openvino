// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor.cpp:462-467 (Tensor::get_ov_constant).
// Encodes the fix: an INT64 *scalar* initializer (shape: []) whose EXTERNAL
// data file/declared length is shorter than one element (e.g. 7 bytes for INT64)
// must be REJECTED with ov::Exception, instead of being silently replaced by a
// zero failsafe constant. Pre-fix this convert_model() succeeds and yields a 0
// scalar; post-fix it throws error::invalid_external_data.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
//
// NOTE: this needs binary fixtures that the source tree does not yet contain:
//   - models/scalar_int64_truncated_external_data.onnx  (scalar initializer,
//     data_location=EXTERNAL, external_data 'length'=7)
//   - the referenced raw weights file containing only 7 bytes
// Hence this is a SKELETON; create the fixtures before enabling.

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scalar_truncated_external_data_throws) {
    // TODO: generate fixture 'scalar_int64_truncated_external_data.onnx' with a
    //       scalar (dims empty) INT64 initializer referencing an external file
    //       whose declared length is 7 (< 8 = sizeof(int64)). The companion raw
    //       data file must contain exactly 7 bytes.
    EXPECT_THROW(convert_model("scalar_int64_truncated_external_data.onnx"),
                 ov::Exception);
}
