// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-822 at
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-300
// where extract_tensor_external_data() blindly casts the protobuf-supplied
// 'offset' integer to uint8_t* (m_tensor_data) when external_data.location ==
// "*/_ORT_MEM_ADDR_/*". A model loaded from a FILE must never be allowed to set
// an ORT_MEM_ADDR pointer; the fixed frontend must reject it (throw) instead of
// dereferencing an attacker-chosen address.
//
// Pre-fix: convert_model() succeeds (or later derefs the bogus pointer -> ASan
//          SEGV / heap-buffer-overflow during constant folding).
// Post-fix: convert_model() throws ov::Exception because ORT_MEM_ADDR is
//          disallowed on the file-load trust boundary.
//
// TODO: this test needs a crafted binary fixture
//   models/ort_mem_addr_external_data.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data = { location:"*/_ORT_MEM_ADDR_/*", offset:"4096", length:"8" }
// Generate it with a small protobuf script mirroring the ONNX TensorProto schema
// (cannot be produced as plain text here).

#include "onnx_utils.hpp"   // TODO: confirm the FrontEndTestUtils helper header used by onnx_import.in.cpp
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TEST mirrors the style of onnx_import.in.cpp (convert_model + EXPECT_THROW).
TEST(onnx_import_external_data, ort_mem_addr_pointer_rejected_on_file_load) {
    // TODO: place crafted fixture under the frontend test models dir and
    //       reference it the same way other onnx_import.in.cpp tests do.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_external_data.onnx"),
                 ov::Exception);
}
