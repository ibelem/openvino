// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 arbitrary read in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:117-129.
// A model whose initializer has data_location=EXTERNAL and an external_data
// 'location' entry equal to the ORT_MEM_ADDR sentinel "*/_ORT_MEM_ADDR_/*",
// with attacker-chosen 'offset' (target virtual address) and 'length', must be
// REJECTED at parse time. Pre-fix: load_external_mem_data() memcpy's from the
// forged pointer (ASan: SEGV / heap-buffer-overflow read). Post-fix: the
// TensorProto constructor throws ov::Exception (invalid_external_data).
//
// NOTE: harness is ov_onnx_frontend_tests (style of onnx_import.in.cpp). This
// needs a crafted protobuf fixture, so it is a SKELETON — the .onnx/.prototxt
// asset must be authored by hand because the malicious sentinel cannot be
// emitted by the standard ONNX exporters.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create the fixture model at
//   onnx/models/external_data/ort_mem_addr_forged.prototxt
// A single Constant/initializer tensor with:
//   data_location: EXTERNAL
//   external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
//   external_data { key: "offset"   value: "<any nonzero decimal addr>" }
//   external_data { key: "length"   value: "16" }
// and reference it from a node so get_ov_constant()/get_external_data() runs.
TEST(onnx_external_data, model_cannot_forge_ort_mem_addr_sentinel) {
    // TODO: confirm convert_model() helper name/signature from onnx_import.in.cpp
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_forged.prototxt"), ov::Exception);
}
