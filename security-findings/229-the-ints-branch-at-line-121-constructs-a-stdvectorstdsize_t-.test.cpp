// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for attribute.hpp:121 (CWE-195 signed->unsigned conversion):
// a negative value in a repeated-int attribute (kernel_shape/strides/dilations)
// must be rejected, not silently wrapped to ~SIZE_MAX and forwarded to ov::Shape.
// Pre-fix: convert_model deserializes -1 as SIZE_MAX -> huge ov::Shape (no throw /
// later oversized-alloc or assert). Post-fix: per-element OPENVINO_ASSERT(v>=0)
// makes convert_model throw ov::Exception at frontend time.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture models/conv_negative_kernel_shape.onnx with a
// Conv node attribute kernel_shape:[-1] (repeated int64). Skeleton below.

OPENVINO_TEST(${BACKEND_NAME}, onnx_conv_negative_kernel_shape_rejected) {
    // TODO: add crafted fixture onnx_import/models/conv_negative_kernel_shape.onnx
    //       containing a Conv node with attribute kernel_shape = [-1] (AttributeProto INTS).
    //       Build it via onnx.helper.make_attribute("kernel_shape", [-1]).
    EXPECT_THROW(convert_model("conv_negative_kernel_shape.onnx"), ov::Exception);
}
