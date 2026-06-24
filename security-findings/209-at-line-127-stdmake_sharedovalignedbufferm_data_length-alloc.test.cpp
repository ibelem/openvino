// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
// (TensorExternalData::load_external_mem_data). Pre-fix: an external_data tensor whose
// `location` == ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") with a huge `length` reaches
// std::make_shared<ov::AlignedBuffer>(m_data_length) with no upper-bound check, throwing
// std::bad_alloc / OOM-killing the process. Post-fix: an out-of-range length must be
// rejected with ov::Exception (error::invalid_external_data) instead of bad_alloc/OOM.
//
// SKELETON: a crafted .onnx fixture is required (proto-level external_data entries cannot be
// expressed inline in this harness), so the symbols/fixture below are placeholders.
#include "onnx_utils.hpp"          // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/test_case.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace from onnx_import.in.cpp

// TODO: create models/external_data/ort_mem_addr_huge_length.onnx :
//   a TensorProto with data_location = EXTERNAL and external_data entries:
//     location = "*/_ORT_MEM_ADDR_/*"   (detail::ORT_MEM_ADDR)
//     offset   = "4096"                 (non-zero so is_valid_buffer is true)
//     length   = "18446744073709551615" (UINT64_MAX -> excessive allocation)
TEST(onnx_external_data, ort_mem_addr_excessive_length_is_rejected) {
    // Pre-fix this constructs AlignedBuffer(UINT64_MAX) -> std::bad_alloc (NOT ov::Exception),
    // i.e. the test fails. Post-fix the length is bounds-checked and converted to ov::Exception.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_huge_length.onnx"), ov::Exception);
}
