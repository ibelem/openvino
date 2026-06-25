// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:126-129 (load_external_mem_data):
//   a file-based ONNX model must NOT be able to spoof the ORT_MEM_ADDR sentinel
//   ("*/_ORT_MEM_ADDR_/*") to coerce reinterpret_cast<char*>(m_offset) + memcpy.
// Pre-fix: convert_model dereferences the attacker offset (ASan: SEGV / heap-buffer-overflow
//   on the memcpy source at tensor_external_data.cpp:129) or crashes.
// Post-fix: the file-based path rejects the spoofed sentinel and throws ov::Exception.
//
// Style mirrors onnx_import.in.cpp in ov_onnx_frontend_tests.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: provide a crafted fixture models/external_data/ort_mem_addr_spoof.onnx where an
//       initializer has data_location=EXTERNAL and external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"   (the ORT_MEM_ADDR sentinel)
//         offset   = <a bogus/non-mappable address, e.g. 0xdeadbeef>
//         length   = <large value, e.g. 0x40000000>
//       A pure-from-disk ONNX model must never reach load_external_mem_data().
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_rejects_spoofed_ort_mem_addr) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_spoof.onnx"), ov::Exception);
}
