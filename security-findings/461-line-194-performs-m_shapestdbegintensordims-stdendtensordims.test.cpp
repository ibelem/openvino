// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195/CWE-190 at tensor.hpp:194 (negative ONNX dim narrowed to SIZE_MAX,
// bypassing the size check at tensor.cpp:467 because SIZE_MAX*SIZE_MAX mod 2^64 == 1).
// Pre-fix: convert_model() on a model whose initializer has dims=[-1,-1], data_type=DOUBLE,
//   and 8 bytes of raw_data builds an ov::op::v0::Constant with shape {SIZE_MAX,SIZE_MAX}
//   (no throw) -> corrupt graph. Post-fix: the non-negative-dim check rejects it with ov::Exception.
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
//
// TODO: provide the crafted fixture models/negative_initializer_dims.onnx (or .prototxt converted
//   to .onnx) with: initializer { dims: -1, dims: -1, data_type: DOUBLE, raw_data: <8 bytes> }.
//   A negative value cannot be expressed in a text .prototxt 'dims' the same way as a normal one;
//   it must be authored as a real proto with a negative int64 dim. This is why a self-contained
//   compile-only test is not achievable here.

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_initializer_dims_rejected) {
    // TODO: confirm CommonTestUtils / util::convert_model helper name used by onnx_import.in.cpp
    EXPECT_THROW(convert_model("negative_initializer_dims.onnx"), ov::Exception);
}