// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-822 in ov::frontend::onnx.
// Pre-fix: convert_model() of a model whose initializer sets
//   external_data location='*/_ORT_MEM_ADDR_/*', offset=<arbitrary addr>, length=N
//   reaches TensorExternalData::load_external_mem_data()
//   (tensor_external_data.cpp:126 reinterpret_cast<char*>(m_offset);
//    tensor_external_data.cpp:129 std::memcpy(dst, addr_ptr, m_data_length))
//   -> arbitrary-address read (ASan: SEGV / heap-buffer-overflow on the bogus pointer).
// Post-fix: the ORT_MEM_ADDR branch at tensor.cpp:455 / tensor.hpp:324 is gated on
//   m_tensor_place != nullptr, so an on-disk file is rejected with ov::Exception
//   (error::invalid_external_data) before any dereference.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO(fixture): add a crafted model 'external_data_ort_mem_addr_ondisk.onnx' under
//   src/frontends/onnx/tests/models/ with a single initializer whose TensorProto has:
//     data_location = EXTERNAL
//     external_data = [ {key:'location', value:'*/_ORT_MEM_ADDR_/*'},
//                       {key:'offset',   value:'4096'},   // any bogus non-zero addr
//                       {key:'length',   value:'4096'} ]
//   (offset/length parsed by std::stoull at tensor_external_data.cpp:24-26).
//   convert_model() resolves model paths via the test models dir (see onnx_utils.hpp).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected_from_disk) {
    // Pre-fix this would dereference the attacker-chosen address inside
    // load_external_mem_data() before this assertion could observe a clean throw.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr_ondisk.onnx"), ov::Exception);
}