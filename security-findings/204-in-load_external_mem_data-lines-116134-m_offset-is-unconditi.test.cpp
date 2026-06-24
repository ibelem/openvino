// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:126-129 (CWE-822/CWE-125).
// A disk/buffer-loaded ONNX model whose initializer carries external_data
// location = "*/_ORT_MEM_ADDR_/*" with an attacker-chosen `offset` must NOT be
// dereferenced as a raw pointer. Pre-fix: tensor.cpp:455 routes to
// load_external_mem_data() purely on string equality and memcpys from the
// attacker address (ASan: heap/unknown-address read or SEGV). Post-fix: the
// ORT_MEM_ADDR branch is gated on in-process provenance (m_tensor_place != null),
// so this proto-only model is rejected with ov::Exception.
//
// TODO(fixture): this test needs a crafted model `ort_mem_addr_external_data.onnx`
//   with a single FLOAT initializer "x" of shape {2} whose TensorProto.external_data =
//   { (location, "*/_ORT_MEM_ADDR_/*"), (offset, "4096"), (length, "8") }.
//   Generate it with onnx.helper (set raw_data empty, data_location=EXTERNAL) and
//   drop it under the onnx frontend test models dir. Replace the path below.
// TODO(symbols): confirm the convert_model helper / models path macro used by the
//   surrounding onnx_import.in.cpp suite (e.g. util::path_join, ONNX_TEST_MODELS).

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // convert_model(...)

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer, reject_ort_mem_addr_external_data_from_file) {
    // Must throw rather than memcpy from the attacker-supplied raw address.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}