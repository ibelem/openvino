// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-822 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (reinterpret_cast<char*>(m_offset) + memcpy, reached via tensor.hpp:322-325).
// A file-loaded ONNX model whose initializer external_data sets
//   location == "*/_ORT_MEM_ADDR_/*" (the ORT_MEM_ADDR marker, hpp:91)
//   with an arbitrary numeric offset/length must be REJECTED, not dereferenced.
// Pre-fix: convert_model attempts memcpy from reinterpret_cast<char*>(offset)
//          -> ASan SEGV / heap-buffer-overflow or process crash.
// Post-fix: the file-loaded TensorProto path refuses ORT_MEM_ADDR and throws ov::Exception.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.
// NOTE: requires a crafted fixture model (see TODO) — marked skeleton accordingly.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/ort_mem_addr_external_data.onnx (or .prototxt) whose single
//       initializer has data_location=EXTERNAL and external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"
//         offset   = "4096"   (an arbitrary, non-mappable address)
//         length   = "256"
//       Use the onnx_import.in.cpp prototxt fixture convention so the importer
//       parses it through the TensorProto ctor (tensor_external_data.cpp:19-36).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_rejects_ort_mem_addr_marker) {
    // The ORT_MEM_ADDR raw-pointer channel must be unreachable from a file-loaded model.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
