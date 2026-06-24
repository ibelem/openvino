// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-20/CWE-822 at:
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:116-129
//   openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:324
// A file-based ONNX model whose external_data 'location' == "*/_ORT_MEM_ADDR_/*"
// and whose 'offset' is an attacker-chosen integer currently flows into
// reinterpret_cast<char*>(m_offset) + memcpy => arbitrary-pointer read.
// Pre-fix: ASan SEGV / heap-buffer-overflow (or silent OOB read) during import.
// Post-fix: import must reject the model with ov::Exception because the
// ORT_MEM_ADDR sentinel is not permitted in a file-based TensorProto.
//
// Harness: ov_onnx_frontend_tests (style of onnx_import.in.cpp), gtest + ASan.
//
// SKELETON: triggering requires a crafted .onnx binary fixture that cannot be
// authored as plain source here.  See TODOs.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: place a crafted model at
//   onnx/frontend/tests/models/external_data/ort_mem_addr_sentinel_rejected.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data: { key:"location", value:"*/_ORT_MEM_ADDR_/*" },
//                  { key:"offset",   value:"4096" },   // arbitrary address
//                  { key:"length",   value:"64" }
// TODO: confirm the exact OPENVINO_TEST namespace/macro and model-loader helper
//       by reading onnx_import.in.cpp in this test tree before use.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_sentinel_rejected) {
    // Pre-fix this convert_model dereferences reinterpret_cast<char*>(4096).
    // Post-fix it must throw because the file-based path may not use ORT_MEM_ADDR.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_sentinel_rejected.onnx"),
                 ov::Exception);
}