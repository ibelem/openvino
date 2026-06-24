// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 untrusted pointer dereference in the ONNX frontend.
// Unchecked code: targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126,129
//   char* addr_ptr = reinterpret_cast<char*>(m_offset); std::memcpy(dst, addr_ptr, m_data_length);
// Reached via targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:455 (and tensor.hpp:324),
// which select load_external_mem_data() by a bare string compare against detail::ORT_MEM_ADDR
// (= "*/_ORT_MEM_ADDR_/*") with no m_tensor_place != nullptr guard.
//
// This test loads a crafted model whose initializer uses data_location=EXTERNAL with
// external_data entries location="*/_ORT_MEM_ADDR_/*", offset=<bogus address>, length=4096.
// PRE-FIX: dispatch enters load_external_mem_data(), reinterpret_casts the attacker offset to a
//          pointer and memcpy-reads -> ASan SEGV / arbitrary read (test crashes, no throw).
// POST-FIX: the file path must NOT reach the ORT_MEM_ADDR branch; convert_model must reject the
//           model with ov::Exception (error::invalid_external_data).
//
// NOTE: this needs a crafted binary .onnx fixture, so it is emitted as a SKELETON.
// Place in the style of openvino/src/frontends/onnx/tests/onnx_import.in.cpp.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // get_models_dir(), convert_model() helpers used across onnx tests
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, reject_ort_mem_addr_in_file_based_model) {
    // TODO: add a crafted fixture model file at
    //   openvino/src/frontends/onnx/tests/models/external_data/ort_mem_addr_arbitrary_read.onnx
    // containing one float initializer with:
    //   tensor.data_location = EXTERNAL
    //   external_data: {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
    //                  {key:"offset",   value:"<decimal bogus virtual address, e.g. 0x4141414141414141>"}
    //                  {key:"length",   value:"4096"}
    // TODO: confirm the exact convert_model() helper name/signature from onnx_utils.hpp in the test tree.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}