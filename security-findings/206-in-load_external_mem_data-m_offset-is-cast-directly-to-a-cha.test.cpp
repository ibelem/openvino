// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (load_external_mem_data raw reinterpret_cast<char*>(m_offset) + memcpy).
//
// Pre-fix: convert_model() of a model whose initializer has
//   external_data location = "*/_ORT_MEM_ADDR_/*", offset = <bogus addr>, length > 0
// reaches load_external_mem_data() (tensor.cpp:455-456 with m_tensor_place==nullptr)
// and memcpy's from an attacker-chosen address -> ASan SEGV / arbitrary read.
// Post-fix: the ORT_MEM_ADDR branch is only reachable when m_tensor_place!=nullptr,
// so a file-sourced proto must throw ov::Exception (invalid_external_data) instead.
//
// NOTE: this requires a crafted .onnx fixture; emitted as a SKELETON.
#include "onnx_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add binary fixture models/ort_mem_addr_file_sourced.onnx whose single
//       initializer carries external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"
//         offset   = "1094795585"   // 0x41414141, an unmapped address
//         length   = "4096"
//       (a file-sourced TensorProto, NOT placed by an in-process ORT session).
TEST(onnx_importer, DISABLED_ort_mem_addr_must_not_dereference_file_sourced_offset) {
    // Must be rejected at parse/convert time, never dereferenced.
    EXPECT_THROW(convert_model("ort_mem_addr_file_sourced.onnx"), ov::Exception);
}
