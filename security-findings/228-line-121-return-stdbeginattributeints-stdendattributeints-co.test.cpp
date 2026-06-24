// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for attribute.hpp:121 (CWE-195 signed->unsigned narrowing of
// AttributeProto::ints() into std::vector<std::size_t>). A model whose MaxPool
// kernel_shape contains a negative element (e.g. [3,-1]) currently turns -1 into
// SIZE_MAX silently; after the fix, get_value<vector<size_t>> must reject the
// negative element so convert_model throws ov::Exception instead of producing a
// bogus SIZE_MAX spatial dim.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// TODO: add the crafted fixture models/negative_kernel_shape.onnx (a MaxPool/Conv
//       node with attribute kernel_shape = [3, -1]) under the frontend test
//       models dir; reuse the existing convert_model() helper from this TU.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_kernel_shape_rejected) {
    // Pre-fix: -1 is narrowed to SIZE_MAX at attribute.hpp:121 and no throw here.
    // Post-fix: the non-negative validation makes import fail cleanly.
    EXPECT_THROW(convert_model("negative_kernel_shape.onnx"), ov::Exception);
}