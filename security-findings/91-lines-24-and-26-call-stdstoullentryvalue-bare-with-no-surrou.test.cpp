// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for unchecked std::stoull at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24 and :26
// (TensorExternalData::TensorExternalData(const TensorProto&)).
//
// What this encodes:
//   A model whose tensor external_data map has a non-numeric 'offset'
//   (e.g. value="not_a_number") or an overflowing 'length'
//   (e.g. value="99999999999999999999999999") must be REJECTED with a
//   frontend ov::Exception (error::invalid_external_data), NOT crash the
//   process with an uncaught std::invalid_argument / std::out_of_range.
//
//   Pre-fix: convert_model throws std::out_of_range/std::invalid_argument
//            (a std::logic_error, not ov::Exception) -> EXPECT_THROW(...,
//            ov::Exception) FAILS (wrong exception type escapes).
//   Post-fix: the constructor rethrows error::invalid_external_data, which
//            derives from ov::Exception -> assertion passes.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_offset_length_rejected) {
    // TODO: provide a crafted ONNX fixture under
    //   src/frontends/onnx/tests/models/external_data/
    // whose initializer has data_location=EXTERNAL and an external_data entry
    //   {key:"offset", value:"not_a_number"}  (or length:"99999999999999999999999999").
    // This binary/protobuf fixture cannot be authored as plain text here, hence
    // the value below is a placeholder file name.
    EXPECT_THROW(convert_model("external_data/external_data_malformed_offset.onnx"),
                 ov::Exception);
}