// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 in Tensor::Tensor (tensor.hpp:197) + get_data_size() (tensor.hpp:375-376).
// A TensorProto with dims:[0], data_type:UNDEFINED (0), non-empty raw_data triggers divide-by-zero
// or uncaught exception when the Tensor constructor fires outside the try/catch at graph.cpp:118.
// Pre-fix: process crashes (SIGFPE from integer divide-by-zero) or throws unhandled exception.
// Post-fix: loading the model throws a controlled ov::Exception (rejected cleanly).

#include <gtest/gtest.h>
#include <fstream>
#include <string>

// ONNX protobuf headers (available in the ov_onnx_frontend_tests build environment)
#include <onnx/onnx_pb.h>

#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"

// Helper: write a minimal ONNX ModelProto to a temp file and return the path.
static std::string write_crafted_onnx(const std::string& path) {
    using namespace ONNX_NAMESPACE;

    ModelProto model;
    model.set_ir_version(7);
    model.set_opset_import(0)->set_version(17);
    model.mutable_opset_import(0)->set_domain("");

    GraphProto* graph = model.mutable_graph();
    graph->set_name("test_graph");

    // Craft the malicious initializer:
    //   dims: [0]         -> m_shape == ov::Shape{0}
    //   data_type: 0      -> UNDEFINED
    //   raw_data: 4 bytes -> non-empty, triggers the division
    TensorProto* init = graph->add_initializer();
    init->set_name("bad_tensor");
    init->add_dims(0);                      // dims: [0]
    init->set_data_type(0);                 // UNDEFINED
    init->set_raw_data("\xDE\xAD\xBE\xEF"); // non-empty raw_data

    // Add a trivial output to keep the graph structurally valid
    // (no nodes; the crash should happen at initializer parse time)
    ValueInfoProto* output = graph->add_output();
    output->set_name("out");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(1); // FLOAT

    std::ofstream ofs(path, std::ios::binary);
    model.SerializeToOstream(&ofs);
    return path;
}

TEST(OnnxImportUndefinedDataType, TensorCtorDivideByZeroUndefinedType) {
    // TODO: adjust tmp_path to a writable scratch directory available in the test environment.
    const std::string tmp_path = "/tmp/crafted_undefined_dtype.onnx";
    write_crafted_onnx(tmp_path);

    ov::Core core;
    // Pre-fix: this call crashes (SIGFPE) or throws std::terminate due to divide-by-zero in
    //   Tensor::get_data_size() at tensor.hpp:375-376 when data_type==UNDEFINED.
    // Post-fix: a controlled ov::Exception is thrown before the division.
    EXPECT_THROW(core.read_model(tmp_path), ov::Exception);
}
