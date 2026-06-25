// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24,126,129
// A file-loaded ONNX model must NOT be able to set external_data location to the
// ORT shared-memory marker "*/_ORT_MEM_ADDR_/*" and have offset reinterpret_cast to a
// raw pointer that is memcpy'd from (tensor_external_data.cpp:126,129).
//
// Pre-fix: convert_model() reaches load_external_mem_data() and performs memcpy from an
//          attacker-chosen address -> ASan SEGV / heap-buffer-overflow read (or arbitrary read).
// Post-fix: the proto path rejects ORT_MEM_ADDR and throws ov::Exception.
//
// This needs a crafted .onnx fixture, so it is a SKELETON: the binary model cannot be
// authored inline without protobuf tooling.

#include "onnx_import_test_helpers.hpp"   // TODO: confirm exact helper header in src/frontends/onnx/tests/
#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: Create models/ort_mem_addr_arbitrary_read.onnx (or models/onnx/ ...) with a
//       TensorProto where:
//         data_location  = EXTERNAL
//         external_data: { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//                        { key:"offset",   value:"<some non-zero integer>" }
//                        { key:"length",   value:"4096" }
//       Place it where the test's model-path resolver (e.g. util::path_join /
//       TEST_ONNX_MODELS_DIRNAME) can find it.
TEST(onnx_external_data, reject_ort_mem_addr_from_file_loaded_model) {
    // TODO: replace with the project's model-loading helper used by onnx_import.in.cpp,
    //       e.g. convert_model("ort_mem_addr_arbitrary_read.onnx").
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
