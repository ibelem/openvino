// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for graph_iterator_proto.cpp:294-300 (CWE-822 untrusted pointer
// dereference). Pre-fix: an EXTERNAL initializer whose external_data location is the
// ORT_MEM_ADDR sentinel and whose offset is an attacker decimal address is accepted
// during a FILE-BASED load and reinterpret_cast to a raw uint8_t* (arbitrary VA read).
// Post-fix: file-based loading must reject the ORT_MEM_ADDR sentinel and throw.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: This requires a crafted .onnx fixture that cannot be expressed inline here,
// so this is a SKELETON. See TODOs.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // ov::frontend::onnx::tests::convert_model

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer, reject_ort_mem_addr_sentinel_on_file_load) {
    // TODO: create models/ort_mem_addr_sentinel.onnx (model.onnx.prototxt) with a single
    //       initializer:
    //         data_location: EXTERNAL
    //         external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
    //         external_data { key: "offset"   value: "4096" }   // attacker target address
    //         external_data { key: "length"   value: "4096" }
    //       and register it with the .in.cpp ${BACKEND_NAME}/model fixture mechanism.
    //
    // Pre-fix: convert_model silently builds a Constant whose data pointer == (uint8_t*)4096
    //          and reading it triggers an ASan/SEGV on an arbitrary VA.
    // Post-fix: extract_tensor_external_data throws because the ORT_MEM_ADDR branch is
    //          gated to trusted in-process callers only.
    EXPECT_THROW(convert_model("ort_mem_addr_sentinel.onnx"), ov::Exception);
}