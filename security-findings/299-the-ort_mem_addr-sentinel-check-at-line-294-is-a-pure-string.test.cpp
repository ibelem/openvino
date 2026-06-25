// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-297
// Pre-fix: a file-loaded ONNX model whose initializer has external_data
// location == "*/_ORT_MEM_ADDR_/*" and an arbitrary 'offset' causes that
// integer to be reinterpret_cast into m_tensor_data (an arbitrary pointer),
// which is dereferenced later during conversion (ASan: SEGV/arbitrary read).
// Post-fix: the file-load path rejects the ORT_MEM_ADDR sentinel and throws.
//
// This test is a SKELETON: it needs a crafted .onnx fixture because the
// trigger is a malformed binary TensorProto, which cannot be expressed in the
// gtest source alone.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: produce models/ort_mem_addr_sentinel.onnx with:
//   graph.initializer[0].data_location = EXTERNAL
//   external_data: {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                  {key:"offset",   value:"0xdeadbeef00"},
//                  {key:"length",   value:"16"}
// Place under the onnx frontend test models dir (see onnx_utils.hpp helpers).
OPENVINO_TEST(${FRONTEND_NAME}_onnx, ort_mem_addr_sentinel_rejected_on_file_load) {
    // TODO: confirm the exact convert_model helper/signature used by this
    // test tree (e.g. convert_model("ort_mem_addr_sentinel.onnx")).
    EXPECT_THROW(convert_model("ort_mem_addr_sentinel.onnx"), ov::Exception);
}
