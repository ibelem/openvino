// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (load_external_mem_data) reached via core/tensor.hpp:324-325.
//
// What this encodes:
//   A file-deserialized ONNX tensor whose external_data 'location' equals the
//   ORT_MEM_ADDR marker ("*/_ORT_MEM_ADDR_/*") with an attacker-chosen 'offset'
//   and 'length' MUST be rejected at parse/convert time. Pre-fix the frontend
//   reinterpret_casts the offset to a pointer and memcpy's length bytes from it
//   (ASan: SEGV / heap-buffer-overflow / wild read). Post-fix (gate tensor.hpp:324
//   on m_tensor_place != nullptr) convert_model() must throw ov::Exception.
//
// NOTE: this requires a crafted model fixture; a self-contained TensorProto built
// inline is not how this test tree drives the frontend, so this is a SKELETON.

#include "onnx_utils.hpp"          // TODO: confirm include used by onnx_import.in.cpp for convert_model()
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: place crafted model under the frontend test models dir, e.g.
//   onnx/models/external_data/ort_mem_addr_wild_pointer.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data: location="*/_ORT_MEM_ADDR_/*", offset="<nonzero decimal addr>", length="4096"
static const std::string MANIFEST_DIR = ""; // TODO: set to ${ONNX_TEST_MODELS} root

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_ext_data_ort_mem_addr_rejected) {
    // TODO: replace with the convert_model helper actually used in onnx_import.in.cpp
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_wild_pointer.onnx"), ov::Exception);
}
