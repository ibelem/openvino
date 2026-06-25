// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-822 untrusted pointer dereference at
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126,129
// (m_offset from a parsed ONNX model is reinterpret_cast<char*> and used as a
// memcpy source). After the fix, a model whose external_data location is the
// ORT_MEM_ADDR sentinel '*/_ORT_MEM_ADDR_/*' with an attacker-chosen offset must
// be rejected during conversion instead of dereferencing the raw address.
//
// Pre-fix: convert_model() reaches load_external_mem_data() which memcpy's from
//          the attacker address -> ASan read of unmapped/out-of-bounds memory or SIGSEGV.
// Post-fix: convert_model() throws ov::Exception (invalid_external_data) because the
//           ORT_MEM_ADDR path is gated to the trusted programmatic constructor only.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_rejected) {
    // TODO: add a crafted fixture model under
    //   src/frontends/onnx/tests/models/external_data/ort_mem_addr_offset.onnx
    // whose single initializer has data_location=EXTERNAL and external_data entries:
    //   key="location" value="*/_ORT_MEM_ADDR_/*"
    //   key="offset"   value="<some bogus address, e.g. 0x4000>"
    //   key="length"   value="4096"
    // (TODO: confirm exact onnx_import test macro/helper names by reading
    //  onnx_import.in.cpp; convert_model resolves the path under models/.)
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_offset.onnx"), ov::Exception);
}