// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (CWE-248 uncaught std::stoull).
// Pre-fix: a non-numeric external_data 'offset' value makes the ctor throw
// std::invalid_argument (NOT ov::Exception) -> EXPECT_THROW(..., ov::Exception) fails.
// Post-fix: ctor rethrows error::invalid_external_data (derives from ov::Exception) -> passes.
//
// NOTE: triggering the path requires a crafted ONNX model whose initializer has
// data_location=EXTERNAL and an external_data entry key="offset", value="abc".
// That is a binary protobuf fixture, so this is a SKELETON with TODOs.

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_nonnumeric_offset_throws_ov_exception) {
    // TODO: provide a crafted model fixture under the onnx frontend models dir, e.g.
    //   models/external_data/external_data_bad_offset.onnx
    // whose tensor has:
    //   data_location: EXTERNAL
    //   external_data { key: "location" value: "data.bin" }
    //   external_data { key: "offset"   value: "abc" }   // non-numeric -> std::stoull throws
    // Use the same convert_model helper as onnx_import.in.cpp.
    EXPECT_THROW(convert_model("external_data/external_data_bad_offset.onnx"),
                 ov::Exception);
    // TODO: add an out-of-range variant (value larger than UINT64_MAX) asserting the
    // same ov::Exception (std::out_of_range pre-fix).
}