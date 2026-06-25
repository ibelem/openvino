// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-129
// A crafted ONNX model whose initializer external_data carries
//   location = "*/_ORT_MEM_ADDR_/*", offset = <arbitrary addr>, length = <n>
// reaches TensorExternalData::load_external_mem_data() via
//   core/tensor.cpp:455-456 and performs reinterpret_cast<char*>(m_offset) + memcpy.
// Pre-fix: arbitrary-address read -> ASan SEGV / heap-buffer-overflow (or silent leak).
// Post-fix: the file-parsed ORT_MEM_ADDR path must be rejected -> ov::Exception
// (error::invalid_external_data) thrown during convert_model.
//
// Style follows onnx_import.in.cpp (ov_onnx_frontend_tests).

#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // provides convert_model(...) used across onnx_import tests

using namespace ov::frontend::onnx::tests;

// TODO(fixture): add a crafted model under
//   src/frontends/onnx/tests/models/ort_mem_addr_external_data.onnx (or .prototxt)
// with an initializer tensor whose external_data entries are:
//   { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//   { key:"offset",   value:"4096" }   // any small non-zero, non-mapped offset
//   { key:"length",   value:"4096" }
// and data_location set so the model loads via the m_tensor_proto branch
// (i.e. NOT through a runtime tensor_place). The model itself must otherwise be
// a minimal valid graph (e.g. a single Identity/Add consuming the initializer).
TEST(onnx_external_data, ort_mem_addr_sentinel_from_file_is_rejected) {
    // Before the fix this convert_model dereferences attacker-controlled m_offset
    // (tensor_external_data.cpp:126) and memcpys length bytes (line 129).
    // After the fix the ORT_MEM_ADDR path must be gated to runtime-constructed
    // TensorExternalData only, so loading from file must throw.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
