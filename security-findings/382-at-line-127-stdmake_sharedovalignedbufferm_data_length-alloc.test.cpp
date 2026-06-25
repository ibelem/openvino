// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 / arbitrary-address read at
// targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
// (and the related memcpy at line 129). A crafted ONNX model whose tensor sets
//   data_location = EXTERNAL,
//   external_data{ location = "*/_ORT_MEM_ADDR_/*", offset = "1",
//                  length = "9223372036854775807" }
// must be rejected (invalid_external_data / ov::Exception) instead of attempting
// an enormous AlignedBuffer allocation + memcpy from a bogus address.
//
// Pre-fix: AlignedBuffer(0x7fffffffffffffff) -> bad_alloc/OOM, or (if it returns)
//          memcpy from addr 0x1 -> ASan/segv. Post-fix: convert_model throws.
//
// Place in: src/frontends/onnx/tests/onnx_import.in.cpp
// TODO: provide the crafted fixture models/ort_mem_addr_huge_length.onnx that
//       carries the EXTERNAL data_location + ORT_MEM_ADDR location with the
//       oversized length string (cannot be generated as a self-contained
//       string literal here — needs a serialized TensorProto).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_huge_length_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_huge_length.onnx"), ov::Exception);
}