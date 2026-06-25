// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 untrusted pointer dereference in
// ov::frontend::onnx::detail::TensorExternalData::load_external_mem_data()
// (tensor_external_data.cpp:126-129) reached via Tensor::get_ov_constant()
// (core/tensor.cpp:455) when a *model-file* tensor sets external_data
// location to the ORT_MEM_ADDR marker "*/_ORT_MEM_ADDR_/*" and an arbitrary
// decimal offset. Pre-fix: m_offset is reinterpret_cast<char*> and memcpy'd,
// triggering an arbitrary read (ASan SEGV / heap-buffer-overflow on read or
// silent info-leak). Post-fix: the ORT_MEM_ADDR branch must be rejected for
// proto-originated tensors (m_tensor_place == nullptr), throwing
// ov::Exception (error::invalid_external_data).
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// This needs a crafted .onnx fixture, so it is a SKELETON.

#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"   // import_onnx_model / convert_model helpers

using namespace ov::frontend::onnx::tests;

// TODO: provide models/ort_mem_addr_arbitrary_offset.onnx:
//   a single Constant/initializer whose TensorProto.external_data is:
//     { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//     { key:"offset",   value:"<arbitrary address, e.g. 140187732541440"> }
//     { key:"length",   value:"4096" }
//   and data_location==EXTERNAL so has_external_data() is true.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_ort_mem_addr_from_model_file_rejected) {
    // TODO: confirm the exact fixture-loading macro/helper name in onnx_import.in.cpp
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_offset.onnx"), ov::Exception);
}
