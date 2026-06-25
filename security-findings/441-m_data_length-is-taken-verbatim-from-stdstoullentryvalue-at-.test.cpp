// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-127
// (load_external_mem_data allocates AlignedBuffer(m_data_length) with no upper bound).
// Pre-fix: a crafted ONNX model with external_data location="*/_ORT_MEM_ADDR_/*" and
//          length=0xFFFFFFFFFFFF0000 reaches Tensor::get_ov_constant -> load_external_mem_data
//          and triggers a ~16EB allocation -> std::bad_alloc escapes the ov::Exception catch
//          at tensor.cpp:479-485 (test process aborts / ASan reports allocation-size-too-big).
// Post-fix: the added size cap throws ov::frontend::onnx::error::invalid_external_data
//          (an ov::Exception), so convert_model rejects the model cleanly.
//
// Style follows src/frontends/onnx/tests/onnx_import.in.cpp (uses convert_model helper
// and the manifest-driven test fixture). A crafted .onnx fixture is required because the
// ORT_MEM_ADDR external_data fields cannot be expressed without a serialized model.
TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    // TODO: provide crafted fixture at
    //   src/frontends/onnx/tests/models/external_data/ort_mem_addr_excessive_length.onnx
    //   containing one initializer with data_location=EXTERNAL and external_data entries:
    //     key="location", value="*/_ORT_MEM_ADDR_/*"
    //     key="offset",   value="<any nonzero, e.g. an arbitrary placeholder address>"
    //     key="length",   value="18446744073709486080"   // 0xFFFFFFFFFFFF0000
    // NOTE: offset is interpreted as a raw pointer; keep length the trigger so the
    //       allocation aborts BEFORE the memcpy at line 129 is reached.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_excessive_length.onnx"),
                 ov::Exception);
}