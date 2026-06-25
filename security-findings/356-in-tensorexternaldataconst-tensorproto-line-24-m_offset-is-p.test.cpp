// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data). Pre-fix: a file-parsed TensorProto whose
// external_data location == ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") with attacker offset is
// reinterpret_cast to a pointer and memcpy'd from -> arbitrary read / SIGSEGV (ASan: SEGV
// or invalid read). Post-fix: the proto-constructed ORT_MEM_ADDR path is rejected with
// ov::Exception (error::invalid_external_data), so convert_model must throw and never deref.
//
// Place in openvino/src/frontends/onnx/tests/onnx_import.in.cpp style; target ov_onnx_frontend_tests.

#include "onnx_utils.hpp"            // TODO: confirm helper header name in the tests/ dir
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add a crafted fixture model file under tests/models/, e.g.
//   external_data_ort_mem_addr.onnx
// containing a single initializer with TensorProto.external_data entries:
//   key="location" value="*/_ORT_MEM_ADDR_/*"
//   key="offset"   value="140737488355328"   // bogus/attacker address
//   key="length"   value="4096"
// (cannot be expressed inline as prototxt safely; must be a binary .onnx fixture)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_is_rejected) {
    // After the fix, the ORT_MEM_ADDR path constructed from a parsed proto (m_tensor_place
    // == nullptr, m_from_proto == true) must be rejected rather than dereferencing m_offset.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr.onnx"), ov::Exception);
}
