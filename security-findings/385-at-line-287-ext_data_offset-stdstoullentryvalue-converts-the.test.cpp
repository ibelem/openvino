// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:287-300
// (extract_tensor_external_data ORT_MEM_ADDR branch).
//
// Pre-fix: a model whose TensorProto has data_location=EXTERNAL with
//   external_data: location="*/_ORT_MEM_ADDR_/*", offset="4096", length="65536"
// causes the frontend to reinterpret_cast(4096) into m_tensor_data and accept it
// (returns true at :300), so convert_model() succeeds and later constant folding
// dereferences the bogus pointer (ASan: SEGV / heap-buffer / unknown-address READ).
// Post-fix (trusted-source gate added before :297): convert_model() must REJECT the
// untrusted ORT_MEM_ADDR model with ov::Exception.
//
// TODO(fixture): add a crafted protobuf model file
//   onnx/models/ort_mem_addr_untrusted.onnx (or .prototxt) containing a single
//   initializer TensorProto with:
//     data_location: EXTERNAL
//     external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
//     external_data { key: "offset"   value: "4096" }
//     external_data { key: "length"   value: "65536" }
//   Place it under the frontend test models dir resolved by
//   util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, ...}).
// TODO(symbols): confirm the exact convert_model/exception helpers from
//   src/frontends/onnx/tests/onnx_import.in.cpp (FrontEndManager / onnx fixture).

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // ov::frontend::onnx::tests::convert_model

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, ort_mem_addr_from_untrusted_model_is_rejected) {
    // Must throw once the ORT_MEM_ADDR branch is gated to trusted callers only.
    EXPECT_THROW(convert_model("ort_mem_addr_untrusted.onnx"), ov::Exception);
}
