// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   (TensorExternalData::load_external_mem_data)
// Pre-fix: a model whose external_data sets location='*/_ORT_MEM_ADDR_/*',
//   offset=1, length=0x7FFFFFFFFFFFFFFF reaches line 127 and calls
//   std::make_shared<ov::AlignedBuffer>(0x7FFFFFFFFFFFFFFF) -> aligned_alloc of
//   ~8EB (aligned_buffer.cpp:18-20, no bound / no null check), then memcpy from
//   reinterpret_cast<char*>(1) at line 129. ASan/the allocator aborts or a null
//   deref occurs.
// Post-fix: load_external_mem_data must reject the oversized / raw-pointer
//   length and the frontend must throw ov::Exception during convert_model.
//
// Harness: ov_onnx_frontend_tests (gtest+ASan), style of onnx_import.in.cpp.
//
// SKELETON: triggering this requires a crafted .onnx fixture (the ORT_MEM_ADDR
// marker + giant length cannot be produced with the in-source model builders
// used elsewhere in onnx_import.in.cpp without a binary fixture), so the model
// file is a TODO.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/external_data/ort_mem_addr_excessive_length.onnx with a
//       TensorProto that has:
//         data_location = EXTERNAL
//         external_data[location] = "*/_ORT_MEM_ADDR_/*"
//         external_data[offset]   = "1"
//         external_data[length]   = "9223372036854775807"  // 0x7FFFFFFFFFFFFFFF
//       and reference it as a Constant/initializer.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_excessive_length.onnx"),
                 ov::Exception);
}
