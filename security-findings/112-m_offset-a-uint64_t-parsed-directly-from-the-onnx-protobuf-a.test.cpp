// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 untrusted pointer dereference in
//   targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24,126,129
//   (TensorExternalData::load_external_mem_data) reached via
//   targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:322-325.
//
// Pre-fix behaviour: a model whose tensor external_data carries
//   location = "*/_ORT_MEM_ADDR_/*", offset = <attacker int>, length = N
// passes the location check (line 117) and the non-zero check (line 121),
// then memcpys N bytes from reinterpret_cast<char*>(offset) (line 129) ->
// ASan heap/wild-read or SIGSEGV.
//
// Post-fix behaviour: the ORT_MEM_ADDR path is rejected when the data
// originated from a parsed protobuf (m_tensor_place == nullptr), so model
// conversion throws ov::Exception instead of dereferencing the address.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
// NOTE: SKELETON — the trigger needs a crafted .onnx fixture that cannot be
// authored as inline C++ here; see TODOs.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_is_rejected) {
    // TODO: add a crafted fixture under frontends/onnx/tests/models/, e.g.
    //   ext_data_ort_mem_addr_untrusted.onnx
    // containing one initializer with:
    //   data_location = EXTERNAL
    //   external_data = { {"location", "*/_ORT_MEM_ADDR_/*"},
    //                     {"offset", "<arbitrary decimal address, e.g. 4096>"},
    //                     {"length", "4096"} }
    // TODO: confirm the exact convert_model() helper signature/model-dir
    //       resolution from onnx_utils.hpp before enabling.
    //
    // Pre-fix: convert_model dereferences the bogus address (ASan abort / SIGSEGV).
    // Post-fix: it must throw because the ORT_MEM_ADDR path is not reachable
    //           from protobuf-parsed external_data.
    EXPECT_THROW(convert_model("ext_data_ort_mem_addr_untrusted.onnx"), ov::Exception);
}