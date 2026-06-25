// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for tensor_external_data.cpp:26 (unbounded std::stoull length)
// and the unchecked sink load_external_mem_data() at cpp:126-129.
// Pre-fix: a parsed ONNX initializer with external_data location='*/_ORT_MEM_ADDR_/*',
//   a forged offset and a huge length reaches reinterpret_cast<char*>(m_offset) + memcpy,
//   producing an arbitrary-address read / std::bad_alloc (ASan: SEGV or allocation-size-too-big).
// Post-fix: the frontend must reject ORT_MEM_ADDR (and/or oversized length) for a
//   file-originated TensorProto, so convert_model throws ov::Exception cleanly.
//
// Style follows onnx_import.in.cpp; lives in the ov_onnx_frontend_tests target.
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

// TODO: add a crafted fixture models/ort_mem_addr_external_length.onnx whose single
//   initializer has data_location=EXTERNAL and external_data entries:
//     location = "*/_ORT_MEM_ADDR_/*"
//     offset   = "4096"                       // forged raw address
//     length   = "18446744073709486080"       // 0xFFFFFFFFFFFF0000, parsed by std::stoull
//   (No raw_data; data_type FLOAT, dims small.) A pure .onnx fixture is required because
//   the trigger is the serialized external_data table, not a runtime API arg.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_length_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_external_length.onnx"), ov::Exception);
}
