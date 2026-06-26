// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data).
//
// Pre-fix: an on-disk .onnx whose tensor has data_location=EXTERNAL with
// external_data entries location="*/_ORT_MEM_ADDR_/*", offset=<attacker addr>,
// length=N reaches tensor.hpp:325 -> load_external_mem_data(), which
// reinterpret_casts the protobuf-supplied offset to char* and memcpy's N bytes
// from an arbitrary address (ASan: SEGV / heap-buffer-overflow read, or silent
// arbitrary read).
// Post-fix: the protobuf parse path rejects ORT_MEM_ADDR (or marks the object as
// protobuf-sourced) and convert_model throws ov::Exception instead.
//
// This test follows the onnx_import.in.cpp style: load a crafted model and
// assert it is rejected. It REQUIRES a crafted fixture, so it is a skeleton.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers used by ov_onnx_frontend_tests

using namespace ov::frontend::onnx::tests;

// TODO: add a crafted fixture, e.g.
//   models/external_data/ort_mem_addr_arbitrary_read.onnx (or .prototxt) with:
//     graph.initializer[0].data_location = EXTERNAL
//     external_data { key:"location" value:"*/_ORT_MEM_ADDR_/*" }
//     external_data { key:"offset"   value:"4096" }   // bogus raw address
//     external_data { key:"length"   value:"4096" }
// matching the naming/loader convention you observe in the existing
// onnx_import_external_data.in.cpp tests (read that file for the exact helper).
TEST(onnx_external_data, DISABLED_ort_mem_addr_marker_rejected_from_protobuf) {
    // TODO: confirm the exact convert_model overload + relative-path helper used
    // by onnx_import_external_data.in.cpp in this tree.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"),
                 ov::Exception);
}
