// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-789 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   auto aligned_memory = std::make_shared<ov::AlignedBuffer>(m_data_length);
// where m_data_length is parsed unchecked from TensorProto.external_data["length"]
// (constructor line 26). The model below sets the external_data location to the
// ORT_MEM_ADDR marker, a nonzero offset, and an enormous length so that
// load_external_mem_data() attempts a ~16 EB allocation.
//
// Pre-fix: the unbounded std::make_shared<ov::AlignedBuffer>(m_data_length) throws
//   std::bad_alloc (DoS) instead of a controlled frontend error.
// Post-fix: an upper-bound/shape check rejects the length and convert_model throws
//   ov::Exception (error::invalid_external_data), which EXPECT_THROW captures.
//
// NOTE: this needs a crafted fixture model "external_data_ort_mem_addr_huge_length.onnx"
// ( deterministic to author by hand: a single Constant/initializer with
//  data_location=EXTERNAL and external_data entries
//  location="*/_ORT_MEM_ADDR_/*", offset="4096", length="18446744073709551600").
// Because that binary fixture and the in-memory pointer semantics are not portable,
// this is emitted as a SKELETON.

#include "onnx_utils.hpp"   // TODO: confirm helper header name in src/frontends/onnx/tests/
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: place crafted fixture under onnx/frontend/tests/models/ and name it here.
static const std::string kModel = "external_data_ort_mem_addr_huge_length.onnx";

TEST(ONNXImportExternalData, rejects_excessive_ort_mem_addr_length) {
    // TODO: convert_model() helper signature mirrors onnx_import.in.cpp; verify name.
    EXPECT_THROW(convert_model(kModel), ov::Exception);
}
