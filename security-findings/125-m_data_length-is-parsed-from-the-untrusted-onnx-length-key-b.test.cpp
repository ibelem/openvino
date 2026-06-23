// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in:
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   (TensorExternalData::load_external_mem_data) — m_data_length parsed unchecked
//   at line 26 and allocated unbounded at line 127.
//
// Encodes the fix: a model whose external_data declares location == ORT_MEM_ADDR
// and an enormous 'length' must be REJECTED (ov::Exception / invalid_external_data)
// rather than attempting a ~16 EB AlignedBuffer allocation.
//
// Pre-fix: ASan/allocator reports the huge allocation request (or std::bad_alloc),
//          OR the arbitrary-pointer memcpy crashes — test does NOT throw ov::Exception cleanly.
// Post-fix: convert_model throws ov::Exception due to the upper-bound check.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// SKELETON: requires a crafted .onnx fixture that cannot be generated read-only.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // FrontEndTestUtils / convert_model helper

using namespace ov::frontend::onnx::tests;

// TODO(fixture): create models/external_data/excessive_mem_length.onnx that contains
//   a TensorProto with data_location = EXTERNAL and external_data entries:
//     {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
//     {key:"offset",   value:"4096"}
//     {key:"length",   value:"18446744073709551615"}   // UINT64_MAX
//   The declared tensor shape/element type must imply a far smaller byte size,
//   so the fix's shape*elt_size bound rejects the oversized 'length'.
//   (A pure-text .onnx cannot be emitted from a read-only tree; hand-build the
//    protobuf with onnx.helper or check in the binary fixture.)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    EXPECT_THROW(convert_model("external_data/excessive_mem_length.onnx"), ov::Exception);
}
