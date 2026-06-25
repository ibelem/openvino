// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// A file-parsed ONNX tensor whose external_data 'location' == "*/_ORT_MEM_ADDR_/*"
// reaches load_external_mem_data(), which reinterpret_casts the file-supplied
// 'offset' to a char* and memcpy's 'length' bytes from it (arbitrary read).
// This test loads a crafted model and asserts the frontend REJECTS it instead
// of dereferencing the attacker-controlled pointer.
//
// Pre-fix: ASan reports a wild/invalid read inside std::memcpy at cpp:129
//          (or the model loads and silently leaks process memory).
// Post-fix: convert_model() throws ov::Exception (invalid_external_data),
//          because the ORT_MEM_ADDR branch is no longer honoured for
//          data parsed from m_tensor_proto.
//
// Style mirrors onnx_import.in.cpp (uses convert_model + EXPECT_THROW).

#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"  // provides convert_model(...) helper used by onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

// TODO(fixture): provide models/external_data/ort_mem_addr_arbitrary_read.onnx
//   A model with a single Constant/initializer tensor where:
//     data_location = EXTERNAL
//     external_data = [ {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                       {key:"offset",   value:"<some nonzero decimal address>"},
//                       {key:"length",   value:"4096"} ]
//   This binary fixture cannot be authored inline here; it must be added to the
//   onnx frontend test models dir (see existing external-data fixtures used by
//   onnx_import.in.cpp). Without it the test cannot run -> skeleton.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_is_rejected) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"),
                 ov::Exception);
}
