// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-300
// (extract_tensor_external_data). An attacker-supplied ONNX model whose tensor
// is data_location=EXTERNAL with external_data {location:"*/_ORT_MEM_ADDR_/*",
// offset:"<decimal addr>", length:"<n>"} currently reaches
//   m_tensor_data = reinterpret_cast<uint8_t*>(stoull(offset));
// with no context gate or validation, planting an arbitrary pointer.
//
// This assertion encodes the fix: a model loaded from a *file* (not an in-process
// ORT hand-off) must NOT honor the ORT_MEM_ADDR marker. Pre-fix the offset is
// cast to a raw pointer and later dereferenced (ASan: SEGV / heap-buffer-overflow
// on read of arbitrary address); post-fix convert_model rejects the model.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected_from_file) {
    // TODO: provide crafted fixture models/external_data/ort_mem_addr_arbitrary.onnx
    //       with a single initializer:
    //         data_location: EXTERNAL
    //         external_data: { key:"location" value:"*/_ORT_MEM_ADDR_/*" }
    //         external_data: { key:"offset"   value:"4096" }   // arbitrary addr
    //         external_data: { key:"length"   value:"64" }
    //       (a protobuf-serialized .onnx must be added under the test data dir;
    //        a self-contained in-memory build is not possible because
    //        extract_tensor_external_data is in an anonymous namespace.)
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary.onnx"),
                 ov::Exception);
}