// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// Pre-fix: load_external_mem_data() reinterpret_casts the attacker-supplied
//   external_data["offset"] integer to char* and memcpy's m_data_length bytes
//   -> ASan: heap/unmapped read or SEGV.
// Post-fix: the ORT_MEM_ADDR path is unreachable from a file-backed TensorProto
//   and convert_model() throws ov::Exception (error::invalid_external_data).
//
// This test needs a crafted .onnx fixture and is therefore a SKELETON: the
// model must contain one initializer with data_location=EXTERNAL and
// external_data { key="location" value="*/_ORT_MEM_ADDR_/*" },
// { key="offset" value="<nonzero>" }, { key="length" value="4096" }.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

// TODO: create the crafted fixture below under
//   openvino/src/frontends/onnx/tests/models/ and add it to the test CMake
//   model list. It must set the first initializer's data_location to EXTERNAL
//   and its external_data entries to location="*/_ORT_MEM_ADDR_/*",
//   offset="4096" (any nonzero), length="4096".
OPENVINO_TEST(${FRONTEND_NAME}, onnx_external_ort_mem_addr_from_file_is_rejected) {
    // Pre-fix this does NOT throw and instead dereferences a bogus pointer
    // (ASan/SEGV); post-fix it must throw because the ORT_MEM_ADDR branch is
    // not allowed for file-backed protos.
    EXPECT_THROW(convert_model("crafted_ort_mem_addr.onnx"), ov::Exception);
}
