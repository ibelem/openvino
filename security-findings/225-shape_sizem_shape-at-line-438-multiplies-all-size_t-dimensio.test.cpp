// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 in onnx frontend tensor.cpp:438-467 / tensor.hpp:194.
// Pre-fix: a crafted ONNX initializer with negative dims (e.g. dims=[-1,-1]) is
// copied into ov::Shape as {SIZE_MAX,SIZE_MAX}; shape_size() wraps to a value that
// matches element_count, defeating the guard at tensor.cpp:467 and constructing an
// ov::op::v0::Constant with a SIZE_MAX shape over a 1-element buffer.
// Post-fix: per-dimension sign/range validation (tensor.hpp:194) or a checked
// shape_size must reject the model with an ov::Exception.
//
// Harness: ov_onnx_frontend_tests (gtest), in the style of onnx_import.in.cpp.
// NOTE: this requires a crafted binary fixture (negative initializer dims) that
// cannot be authored as plain text here -> emitted as a SKELETON.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: provide models/initializer_negative_dims.onnx containing an initializer
//       whose TensorProto.dims = [-1, -1] with raw_data holding exactly 1 float.
//       Build it with onnx.helper / a hex editor; place under the frontend test
//       models dir consumed by convert_model().
OPENVINO_TEST(${BACKEND_NAME}, onnx_initializer_negative_dims_rejected) {
    // TODO: confirm convert_model() helper signature in onnx_utils.hpp for this tree.
    EXPECT_THROW(convert_model("initializer_negative_dims.onnx"), ov::Exception);
}
