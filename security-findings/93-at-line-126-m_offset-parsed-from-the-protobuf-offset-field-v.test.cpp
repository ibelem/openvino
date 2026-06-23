// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (load_external_mem_data reinterpret_casts the model-controlled offset to a
//  pointer and memcpy's m_data_length bytes from it).
//
// What this encodes: a file-based ONNX model whose external_data uses
//   location = "*/_ORT_MEM_ADDR_/*", offset = <arbitrary integer>, length = <n>
// must be REJECTED by the frontend instead of dereferencing the integer as a
// pointer. Pre-fix: under ASan this aborts with a SEGV / heap-buffer-overflow
// (or wild read) in load_external_mem_data. Post-fix: convert_model throws
// ov::Exception (error::invalid_external_data) before any memcpy.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
//
// SKELETON: building the crafted fixture by hand is required because
// ORT_MEM_ADDR normally only originates from the trusted in-process path, never
// from a serialized model on disk, so no existing test .onnx exercises it.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted model file under the frontend test models dir, e.g.
//   onnx/external_data/ort_mem_addr_arbitrary_offset.onnx
// whose single initializer has data_location=EXTERNAL and external_data:
//   key="location" value="*/_ORT_MEM_ADDR_/*"
//   key="offset"   value="0xdeadbeef"   (any non-zero integer)
//   key="length"   value="64"
// (Author with onnx.helper / protobuf text; commit as a binary fixture.)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected) {
    // TODO: confirm exact helper name (convert_model vs. import_onnx_model)
    //       from onnx_import.in.cpp in this checkout.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_offset.onnx"),
                 ov::Exception);
}