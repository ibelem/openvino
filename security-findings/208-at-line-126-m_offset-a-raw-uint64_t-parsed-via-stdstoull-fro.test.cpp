// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-822/CWE-125 at
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129.
// A file-loaded ONNX model whose initializer has data_location=EXTERNAL and an
// external_data entry {location: "*/_ORT_MEM_ADDR_/*", offset: <addr>, length: <n>}
// must be REJECTED at parse time (ctor) rather than reaching the
// reinterpret_cast<char*>(m_offset)+memcpy. Pre-fix this either segfaults
// (unmapped addr) or performs an OOB read flagged by ASan; post-fix the
// frontend throws ov::Exception during convert_model.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected) {
    // TODO: provide crafted fixture 'external_data/ort_mem_addr_offset.onnx':
    //   one initializer tensor, data_location = EXTERNAL, and external_data:
    //     location = "*/_ORT_MEM_ADDR_/*"
    //     offset   = "<some nonzero decimal address>"
    //     length   = "<nonzero, e.g. 64>"
    //   (cannot be inlined as text — ONNX frontend tests load a .onnx/.prototxt
    //    fixture from models dir; add under onnx/tests/models/external_data/.)
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_offset.onnx"), ov::Exception);
}