// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   reached via openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:453-456
//
// A disk-sourced TensorProto whose external_data 'location' equals the ORT_MEM_ADDR
// marker "*/_ORT_MEM_ADDR_/*" with an attacker-chosen decimal 'offset' must NOT be
// dereferenced. Pre-fix, convert_model() reaches load_external_mem_data(), casts
// the offset to char* and memcpy's 'length' bytes from it (ASan: SEGV / heap-buffer
// read on an unmapped/arbitrary address). Post-fix, get_ov_constant() rejects the
// ORT_MEM_ADDR path when m_tensor_place == nullptr and throws ov::Exception
// (error::invalid_external_data).
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// TODO(fixture): add a crafted model file
//   onnx/models/ort_mem_addr_arbitrary_read.onnx (or generate it in-test via the
//   ONNX protobuf API) containing one initializer with:
//     data_location = EXTERNAL
//     external_data = { {"location", "*/_ORT_MEM_ADDR_/*"},
//                       {"offset", "<some decimal VA, e.g. 4096>"},
//                       {"length", "<bytes matching the declared shape, e.g. 16>"} }
//     dims/data_type chosen so element_count == shape_elements (tensor.cpp:467).
//   I cannot hand-author the binary fixture here, so the symbol name below is a
//   placeholder — confirm the exact file name once the fixture is committed.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_disk_rejected) {
    // Must throw rather than dereference the attacker-controlled offset.
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
