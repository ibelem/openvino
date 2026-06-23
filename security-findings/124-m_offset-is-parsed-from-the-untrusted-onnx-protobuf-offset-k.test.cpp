// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (load_external_mem_data: reinterpret_cast<char*>(m_offset) + memcpy)
// reached via core/tensor.cpp:455-456 when a disk-loaded TensorProto sets
//   external_data location == "*/_ORT_MEM_ADDR_/*" and a non-zero offset.
//
// The assertion encodes the fix: a model whose initializer names the
// ORT_MEM_ADDR marker with an attacker-controlled offset MUST be rejected
// (ov::Exception / invalid_external_data) at convert_model time, never
// dereferenced. Pre-fix this test crashes under ASan (wild memcpy from
// reinterpret_cast<char*>(0x1000)); post-fix it throws and the EXPECT_THROW
// passes.
//
// NOTE (fallback skeleton): triggering the path requires a crafted .onnx
// fixture that carries external_data {location="*/_ORT_MEM_ADDR_/*",
// offset="4096", length="65536"} on an initializer. That binary fixture does
// not yet exist in the test data tree, so the model name below is a TODO.
OPENVINO_TEST(${BACKEND_NAME}, onnx_ort_mem_addr_offset_rejected) {
    // TODO: add a crafted model under onnx/frontend/tests/models/, e.g.
    //   external_data_ort_mem_addr_arbitrary_offset.onnx
    // It must contain one initializer with data_location=EXTERNAL and
    // external_data entries:
    //   {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
    //   {key:"offset",   value:"4096"}
    //   {key:"length",   value:"65536"}
    // Pre-fix: convert_model dereferences addr 0x1000 -> ASan SEGV.
    // Post-fix: the ORT_MEM_ADDR branch is refused for TensorProto-derived
    //           tensors (m_tensor_place == nullptr) and an ov::Exception is
    //           thrown.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr_arbitrary_offset.onnx"),
                 ov::Exception);
}