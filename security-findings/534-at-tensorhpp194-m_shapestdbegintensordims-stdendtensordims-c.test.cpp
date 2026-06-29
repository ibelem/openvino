// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor.hpp:194 signed-to-unsigned conversion of negative TensorProto dims.
// A crafted TensorProto with dims: [-1, 3] must be rejected at model-load time (pre-fix: silently
// wraps to SIZE_MAX; post-fix: FRONT_END_GENERAL_CHECK fires and throws ov::AssertFailure).

#include <gtest/gtest.h>

// Standard OV ONNX import helpers (mirror onnx_import.in.cpp style)
#include "onnx_test_util.hpp"   // provides convert_model / create_model helpers
#include "openvino/frontend/exception.hpp"

// Include ONNX protobuf to build the crafted model in-memory
#include <onnx/onnx_pb.h>

namespace {

// Serialise a minimal ONNX ModelProto whose sole initializer has a negative dim.
std::string make_negative_dim_onnx() {
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("negative_dim_test");

    // Initializer with dims: [-1, 3] — negative first dimension
    auto* init = graph->add_initializer();
    init->set_name("bad_tensor");
    init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    init->add_dims(-1);  // <-- negative dim: triggers the bug
    init->add_dims(3);
    // Add 3 float values so the element count is 3 (will not match SIZE_MAX*3 mod 2^64)
    init->add_float_data(1.0f);
    init->add_float_data(2.0f);
    init->add_float_data(3.0f);

    // Minimal output so the graph is syntactically valid
    auto* out = graph->add_output();
    out->set_name("bad_tensor");
    auto* ttype = out->mutable_type()->mutable_tensor_type();
    ttype->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    std::string buf;
    model.SerializeToString(&buf);
    return buf;
}

} // namespace

// Pre-fix: convert_model succeeds (silently corrupts shape) or throws invalid_external_data
// (still a DoS, but shape_size overflow is the root cause).
// Post-fix: must throw an ov::Exception / AssertFailure before m_shape is populated.
TEST(OnnxNegativeDimRegression, NegativeDimIsRejectedAtLoad) {
    const std::string model_bytes = make_negative_dim_onnx();
    // Write to a temp file and load via the ONNX frontend
    const std::string tmp_path = testing::TempDir() + "/negative_dim.onnx";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write(model_bytes.data(), static_cast<std::streamsize>(model_bytes.size()));
    }
    // Expect an OV exception (FRONT_END_GENERAL_CHECK violation) on model load.
    // Pre-fix this may throw a different error or even succeed (corrupted shape propagates).
    EXPECT_THROW(
        ov::frontend::onnx::tests::convert_model(tmp_path),
        ov::AssertFailure
    );
}
