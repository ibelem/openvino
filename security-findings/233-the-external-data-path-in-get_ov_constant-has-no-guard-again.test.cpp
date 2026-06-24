// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-843 type confusion in
//   openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:480 (Tensor::get_ov_constant)
// Unchecked path: a STRING initializer with external_data reaches the
//   Constant(element::string, shape, AlignedBuffer) ctor (constant.cpp:301) with
//   raw external-file bytes, which are later interpreted as std::string internals.
// The fix adds: if (ov_type == ov::element::string) throw error::invalid_external_data(...);
//
// This test loads a crafted ONNX model whose single initializer is type=STRING (data_type=8),
// shape=[1], with data_location pointing to an external file of exactly sizeof(std::string)
// bytes. Pre-fix: model loads (or ASan reports a heap overflow when the bytes are read as
// std::string). Post-fix: convert_model throws ov::Exception ("External string tensors are
// not supported").
//
// Harness: ov_onnx_frontend_tests (style of onnx_import.in.cpp).
//
// TODO(fixture): add the crafted model + external file under
//   onnx/frontend/tests/models/, e.g. external_data/string_external_data.onnx
//   referencing a sibling raw file of sizeof(std::string) arbitrary bytes.
//   These binary fixtures cannot be generated here.
// TODO(symbols): confirm the test helper name (FrontEndTestUtils::convert_model /
//   onnx_import test util) and the OPENVINO_TEST(${BACKEND_NAME}, ...) macro actually used
//   in this test tree before committing.

OPENVINO_TEST(${FRONTEND_NAME}_onnx, string_tensor_with_external_data_is_rejected) {
    // TODO: use the project's convert_model helper used elsewhere in onnx_import.in.cpp
    EXPECT_THROW(
        convert_model("external_data/string_external_data.onnx"),
        ov::Exception);
}