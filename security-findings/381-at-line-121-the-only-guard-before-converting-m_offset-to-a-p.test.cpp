// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-129
// (TensorExternalData::load_external_mem_data). A crafted ONNX model whose constant
// tensor declares data_location=EXTERNAL with external_data
//   location = "*/_ORT_MEM_ADDR_/*", offset = <attacker addr>, length = <N>
// reaches load_external_mem_data() via core/tensor.hpp:324 and performs
//   memcpy(dst, reinterpret_cast<char*>(offset), N) — an arbitrary in-process read.
// PRE-FIX: the model is accepted and the cast/memcpy runs on the attacker offset
//          (ASan: SEGV / heap-buffer-overflow on read, or silent info-leak).
// POST-FIX: the frontend must reject an ORT_MEM_ADDR location that came from the
//           parsed protobuf (not from the trusted in-memory ctor), so convert_model throws.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// NOTE: needs a crafted .onnx fixture; symbol names below mirror the existing suite
// but the fixture path/loader must be confirmed against the local test tree.
#include "onnx_utils.hpp"   // TODO: confirm helper header providing convert_model()
#include "common_test_utils/test_case.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add models/ort_mem_addr_arbitrary_read.onnx — a model with one Constant/initializer
//       tensor: data_location=EXTERNAL, external_data=[
//         {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//         {key:"offset",   value:"<nonzero decimal, e.g. a bogus address>"},
//         {key:"length",   value:"4096"}]
TEST(onnx_importer, ort_mem_addr_offset_from_model_is_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
