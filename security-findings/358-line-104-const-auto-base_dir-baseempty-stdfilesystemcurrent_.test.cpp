// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for file_util.cpp:104 / tensor_external_data.cpp:47,77 (work item 358):
// when an ONNX model is parsed from memory (empty model_dir) and a tensor declares
// external_data with a relative 'location', the loader MUST reject it instead of
// silently resolving against std::filesystem::current_path().
// Pre-fix: sanitize_path falls back to CWD and the read succeeds (no throw),
//          confirming loss of the model-dir containment invariant.
// Post-fix: load_external_data throws ov::Exception / invalid_external_data.
//
// Target test binary: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
//
// NOTE: this needs a crafted .onnx whose initializer uses data_location=EXTERNAL
// and external_data[location] pointing at a file under CWD, fed through an
// in-memory stream (no model path) so get_model_dir() == {}. The model bytes are
// not available here, so this is a SKELETON.

#include <fstream>
#include <sstream>
#include "common_test_utils/test_assertions.hpp"
#include "onnx_utils.hpp"  // FrontEndTestUtils / load_from_stream helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer_external_data, reject_external_data_when_model_dir_empty) {
    // TODO: replace with a real serialized ONNX proto (in-memory, no file path) whose
    //       single initializer has data_location = EXTERNAL and
    //       external_data = { {"location", "some_relative_file.bin"}, {"length","4"} }.
    //       Build it via onnx::ModelProto and SerializeToString so model_dir is empty.
    std::string model_bytes = /* TODO: crafted_external_data_model() */ "";
    ASSERT_FALSE(model_bytes.empty()) << "TODO: supply crafted in-memory ONNX model";

    std::istringstream model_stream(model_bytes);

    // TODO: confirm the exact helper that reads a model from a stream with NO path
    //       (e.g. ov::Core::read_model(stream) or FrontEnd::load(stream)) in this tree.
    // Pre-fix this either succeeds or reads CWD/some_relative_file.bin; post-fix it
    // must throw because model_dir is empty for an in-memory model.
    OV_EXPECT_THROW(/* auto model = */ load_model_from_stream(model_stream),
                    ov::Exception,
                    testing::HasSubstr("external data"));
}
