// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 untrusted pointer dereference.
// Unchecked code: openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   char* addr_ptr = reinterpret_cast<char*>(m_offset);  // m_offset from model 'offset' (line 24)
//   std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length);
// Reached from core/tensor.cpp:455 / core/tensor.hpp:324 when data_location()=="*/_ORT_MEM_ADDR_/*".
//
// What this encodes: a model whose initializer has data_location=EXTERNAL and
// external_data { location="*/_ORT_MEM_ADDR_/*", offset=<garbage addr>, length=N }
// MUST be rejected (throw ov::Exception / error::invalid_external_data) instead of
// dereferencing the attacker-chosen address. Pre-fix: ASan SEGV / heap-buffer-overflow
// or arbitrary read inside load_external_mem_data. Post-fix: convert_model throws
// because the ORT_MEM_ADDR branch is gated on m_tensor_place != nullptr.
//
// FALLBACK SKELETON: this needs a crafted .onnx fixture that cannot be produced from
// the read-only tree, so symbol-exact wiring of the fixture is left as TODO.

#include <gtest/gtest.h>
#include "onnx_utils.hpp"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: add a crafted fixture model under the onnx frontend test models dir, e.g.
//   external_data_ort_mem_addr_untrusted.onnx
// containing a single tensor initializer:
//   data_location: EXTERNAL
//   external_data: { key:"location" value:"*/_ORT_MEM_ADDR_/*" }
//                  { key:"offset"   value:"4096" }   // bogus, non-owned address
//                  { key:"length"   value:"64"   }
// (Must be referenced by the test manifest the same way other onnx_import models are.)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_must_be_rejected) {
    // Pre-fix this either reads/segfaults at tensor_external_data.cpp:129; post-fix it throws.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr_untrusted.onnx"), ov::Exception);
}
