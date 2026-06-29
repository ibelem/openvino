// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for: tensor.hpp:197 / graph.cpp:118
// Flaw: Tensor constructor (called outside try/catch at graph.cpp:118) calls
//       get_data_size() which calls get_onnx_data_size(data_type=UNDEFINED=0)
//       at tensor.hpp:375-376, triggering an uncaught ov::Exception.
// Pre-fix: Graph::Graph() propagates the exception → import_onnx_model() throws.
// Post-fix: constructor call is inside try/catch with 'continue'; loading succeeds.
//
// Build target: ov_onnx_frontend_tests
// Filter:       --gtest_filter=OnnxImportRegression.UndefinedDtypeRawDataInitializer
// Expected sanitizer behavior (pre-fix): ov::Exception propagates uncaught from
//   Graph::Graph(), causing std::terminate or unhandled exception at call site.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "onnx/onnx_pb.h"
#include "onnx_import/onnx_import.hpp"
#include "openvino/frontend/exception.hpp"

namespace {
std::filesystem::path make_temp_model(const std::string& name) {
    // Build a minimal ONNX ModelProto with one initializer:
    //   data_type = 0 (UNDEFINED), dims = [0], raw_data = "\x00" (1 byte)
    // The combination dims=[0] => m_shape==ov::Shape{0} triggers the scalar
    // heuristic at tensor.hpp:197 which evaluates get_data_size(), reaching
    // the raw-data branch at tensor.hpp:375-376 and calling
    // get_onnx_data_size(UNDEFINED) which throws.
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(17);

    auto* graph = model.mutable_graph();
    graph->set_name("regression_533");

    // Malformed initializer
    auto* init = graph->add_initializer();
    init->set_name("bad_init");
    init->set_data_type(0);       // UNDEFINED
    init->add_dims(0);            // dims=[0] => ov::Shape{0}
    init->set_raw_data("\x00", 1); // has_raw_data() == true

    // Minimal graph output referencing the initializer so parsing proceeds
    auto* out = graph->add_output();
    out->set_name("bad_init");
    auto* ttype = out->mutable_type()->mutable_tensor_type();
    ttype->set_elem_type(1);  // FLOAT (placeholder)
    ttype->mutable_shape();   // empty shape

    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream ofs(path, std::ios::binary);
    EXPECT_TRUE(model.SerializeToOstream(&ofs));
    ofs.close();
    return path;
}
}  // namespace

TEST(OnnxImportRegression, UndefinedDtypeRawDataInitializer) {
    // Pre-fix:  this throws an ov::Exception escaping Graph::Graph().
    // Post-fix: the constructor call is inside the try/catch with 'continue';
    //           the bad initializer is skipped and loading succeeds (EXPECT_NO_THROW).
    const auto model_path = make_temp_model("regression_533_undefined_dtype.onnx");

    // After the fix, model loading must not throw.
    // (If the test fails with ov::Exception, the fix has not been applied.)
    EXPECT_NO_THROW({
        ov::frontend::onnx::import_onnx_model(model_path.string());
    });

    std::filesystem::remove(model_path);
}