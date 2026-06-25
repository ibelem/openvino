// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for utils.cpp:107 (ov::frontend::onnx::common::onnx_to_ov_shape).
// Encodes the fix: a TensorShapeProto dim carrying a negative static dim_value
// (other than the -1 OpenVINO-dynamic convention, which ONNX never uses for
// dim_value) must be REJECTED, not silently clipped to a static Dimension(0)
// by Interval::canonicalize() (interval.cpp:47-54).
//
// Pre-fix: onnx_to_ov_shape returns PartialShape{0} for dim_value=-2 (no throw)
//          -> EXPECT_THROW fails.
// Post-fix (OPENVINO_ASSERT(dim_val >= 0, ...)): the call throws ov::Exception
//          -> EXPECT_THROW passes.
//
// NOTE: onnx_to_ov_shape takes onnx::TensorShapeProto, which requires linking
// the onnx_common protobuf headers. If those symbols are not exposed to the
// onnx_import test target, fall back to the convert_model("crafted.onnx")
// pattern of onnx_import.in.cpp with a fixture whose value_info encodes
// dim_value: -2. TODOs below mark what must be confirmed against the tree.

#include <gtest/gtest.h>
#include "openvino/core/except.hpp"

// TODO: confirm the public header that declares onnx_to_ov_shape and pulls in
//       the generated onnx.proto TensorShapeProto (e.g. "onnx_common/utils.hpp").
// #include "onnx_common/utils.hpp"

using namespace ov::frontend::onnx::common;

TEST(onnx_common_utils, onnx_to_ov_shape_rejects_negative_dim_value) {
    // TODO: replace with the real generated protobuf type / namespace.
    // ::onnx::TensorShapeProto shape;
    // auto* d = shape.add_dim();
    // d->set_dim_value(-2);  // attacker-controlled negative static dim
    //
    // // Pre-fix this silently yields PartialShape{0}; post-fix it must throw.
    // EXPECT_THROW(onnx_to_ov_shape(shape), ov::Exception);
    GTEST_SKIP() << "TODO: wire up TensorShapeProto include for onnx_common test target";
}
