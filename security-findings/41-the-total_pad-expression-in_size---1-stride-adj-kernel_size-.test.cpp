// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 signed-integer overflow in
//   openvino/src/frontends/onnx/frontend/src/op/conv_transpose.cpp:218-219
// Pre-fix: convert_model on a ConvTranspose with strides=[4611686018427387905]
//          and output_shape=[1] over a static [1,1,3] input triggers signed
//          overflow (UBSan: "signed integer overflow" in conv_transpose) or a
//          garbage pad fed to ConvolutionBackpropData.
// Post-fix: the frontend rejects the out-of-range stride and throws ov::Exception.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan/UBSan), style of onnx_import.in.cpp.
// NOTE: this requires a crafted model fixture. The exact strides value cannot be
// expressed in the prototxt the auto-generator emits without a real serialized
// model, so a binary .onnx fixture is needed -> see TODOs below.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: place a crafted model at
//   openvino/src/frontends/onnx/tests/models/conv_transpose_stride_overflow.onnx
// containing a single ConvTranspose node with:
//   input  X : tensor<float>[1,1,3]
//   input  W : tensor<float>[1,1,1]
//   attr   strides    = [4611686018427387905]   (2^62 + 1)
//   attr   output_shape = [1]
// (output_shape must be non-empty and shapes static so the line-203 branch runs).
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv_transpose_stride_overflow) {
    // Pre-fix: signed-overflow UB at conv_transpose.cpp:218-219 (UBSan trap) or a
    // bogus graph. Post-fix: CHECK_VALID_NODE rejects the out-of-range stride.
    EXPECT_THROW(convert_model("conv_transpose_stride_overflow.onnx"), ov::Exception);
}
