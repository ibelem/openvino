// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-125 OOB/arbitrary read at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// A file-parsed ONNX TensorProto with external_data location == "*/_ORT_MEM_ADDR_/*"
// and a nonzero offset/length must NOT be allowed to reach load_external_mem_data()
// (which memcpy's from reinterpret_cast<char*>(offset)). After the fix, importing
// such a model must throw instead of dereferencing an attacker-chosen address.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
//
// NOTE: This needs a crafted .onnx fixture (initializer with data_location=EXTERNAL,
//       external_data{location:"*/_ORT_MEM_ADDR_/*", offset:"<addr>", length:"4096"}).
//       Pre-fix this either reads arbitrary memory or SIGSEGVs (ASan-detectable);
//       post-fix convert_model throws. Marked skeleton because the binary fixture
//       and exact convert_model helper must be supplied/verified against the tree.

#include "onnx_utils.hpp"   // TODO: confirm helper header providing convert_model()
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add models/external_data_ort_mem_addr.onnx fixture encoding:
//   initializer { data_location: EXTERNAL,
//                 external_data: [ {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                                  {key:"offset",   value:"<some nonzero decimal>"},
//                                  {key:"length",   value:"4096"} ] }
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected) {
    // Must throw: file-parsed proto must never use an in-process memory address.
    EXPECT_THROW(convert_model("external_data/external_data_ort_mem_addr.onnx"),
                 ov::Exception);
}
