// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data) reached via
//   core/tensor.hpp:324 -> get_external_data() for a file-sourced TensorProto
//   whose external_data 'location' == "*/_ORT_MEM_ADDR_/*".
//
// Pre-fix: convert_model() on the crafted model reaches
//   reinterpret_cast<char*>(m_offset) + memcpy(...) -> arbitrary-address read
//   (ASan: SEGV / heap-buffer-overflow / use-after-poison on the wild address).
// Post-fix: the ORT_MEM_ADDR location parsed from a deserialized TensorProto is
//   rejected, so convert_model throws ov::Exception and no wild read occurs.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// NOTE: this needs a crafted binary .onnx fixture; emitted as a SKELETON.

#include "gtest/gtest.h"
#include "common_test_utils/test_constants.hpp"
#include "onnx_utils.hpp"  // for convert_model / FrontEndTestUtils helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, reject_ort_mem_addr_location_from_file) {
    // TODO: add fixture models/external_data/ort_mem_addr_marker.onnx with ONE
    //       initializer carrying data_location=EXTERNAL and external_data entries:
    //         {key="location", value="*/_ORT_MEM_ADDR_/*"}
    //         {key="offset",   value="<some non-zero address, e.g. 0xdeadbeef>"}
    //         {key="length",   value="4096"}
    //       (Cannot author a binary protobuf fixture in a read-only tree.)
    //
    // EXPECT_THROW encodes the fix: a file-sourced ORT_MEM_ADDR marker must be
    // rejected instead of being reinterpreted as a raw pointer.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_marker.onnx"), ov::Exception);
}
