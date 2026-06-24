// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data), reachable via core/tensor.hpp:322-325.
// Pre-fix: loading a model whose initializer has data_location=EXTERNAL and
//   external_data{location="*/_ORT_MEM_ADDR_/*", offset=<addr>, length=N} causes
//   reinterpret_cast<char*>(m_offset) + memcpy => arbitrary-address read (ASan: SEGV/heap-buffer-overflow).
// Post-fix: the file-sourced ORT_MEM_ADDR sentinel must be rejected with ov::Exception
//   (error::invalid_external_data), so convert_model() throws instead of dereferencing.
//
// Lives in: openvino/src/frontends/onnx/tests/onnx_import.in.cpp style file,
// built as target ov_onnx_frontend_tests.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // get_model_path / convert_model helpers

using namespace ov::frontend::onnx::tests;

// TODO: Provide a crafted fixture model 'ort_mem_addr_external_data.onnx' under
//       onnx/models/ whose single initializer has:
//         data_location = EXTERNAL
//         external_data: location="*/_ORT_MEM_ADDR_/*", offset="4096", length="4096"
//       (offset is a bogus low address; ANY non-zero offset triggers the cast+memcpy pre-fix).
//       A protobuf builder cannot be inlined here without the ONNX_NAMESPACE headers and
//       a serialize step, so the binary fixture is required.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
