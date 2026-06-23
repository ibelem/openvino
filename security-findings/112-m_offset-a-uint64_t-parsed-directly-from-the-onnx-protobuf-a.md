# Security finding #112: m_offset (a uint64_t parsed directly from the ONNX protobuf at line…

**Summary:** m_offset (a uint64_t parsed directly from the ONNX protobuf at line…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary out-of-process-heap memory read: the memcpy at line 129 copies `m_data_length` bytes from an attacker-controlled address into an AlignedBuffer that is then returned as model constant data. On 64-bit hosts this allows reading any mapped virtual address — stack, heap, code, secrets — and returning it as model weights (information leak). With a carefully chosen address it can also cause a segfault (DoS). Affects any caller that loads an ONNX model with external data through this code path.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX protobuf external_data fields parsed in TensorExternalData::TensorExternalData(const TensorProto&) at line 19-30

## Description / Root cause
m_offset (a uint64_t parsed directly from the ONNX protobuf at line 24 via std::stoull) is cast to a raw char* pointer at line 126 (`char* addr_ptr = reinterpret_cast<char*>(m_offset)`) and then used as the source of memcpy at line 129. The only guard (lines 121-124) checks that m_offset is non-zero, not that it points to valid or safe memory. An attacker who sets `location = "*/_ORT_MEM_ADDR_/*"` and supplies an arbitrary 64-bit value for `offset` will pass the check at line 117 and the non-zero check at line 121, causing memcpy to read from the attacker-specified address.

**Validator analysis:** VALIDATED for openvino. The vulnType CWE-822 (Untrusted Pointer Dereference) is accurate: m_offset is parsed via std::stoull directly from protobuf external_data 'offset' (tensor_external_data.cpp:24), cast to a raw char* (line 126) and used as a memcpy source (line 129). The only guard (lines 121-124) verifies m_offset is non-zero and m_data_length sane — it never validates the address points to safe/owned memory. Reachability is confirmed: tensor.hpp get_external_data() (lines 316-332) is the standard constant-loading path; on the ordinary protobuf path (m_tensor_place==nullptr) it constructs TensorExternalData(*m_tensor_proto) and dispatches to load_external_mem_data() whenever the file-supplied location string equals '*/_ORT_MEM_ADDR_/*'. Nothing restricts this branch to the internal TensorONNXPlace constructor (line 37/319), so a plain convert_model(crafted.onnx) reaches it. The stated impact (arbitrary in-process memory read returned as model weights, or segfault DoS) is accurate, though it is an in-process — not 'out-of-process-heap' — read; the leaked bytes become a Constant's data and can be exfiltrated via model outputs. The proposed fix is correct and sufficient in principle: the ORT_MEM_ADDR/raw-address path must only be honored when the data originates from the internal TensorONNXPlace path, not from protobuf-parsed fields. Concretely, gate at tensor.hpp:324 so load_external_mem_data() is only invoked when m_tensor_place != nullptr (i.e., the address came from ORT in-memory, not the file), and/or add the m_from_trusted_source flag set solely by the size_t-offset constructor (line 37) with load_external_mem_data() throwing error::invalid_external_data when the flag is unset. Either fully closes the untrusted-file vector while preserving the legitimate ORT EP use.

## Exploit / Proof of Concept
Craft an ONNX model protobuf with a tensor whose external_data entries include `key="location" value="*/_ORT_MEM_ADDR_/*"`, `key="offset" value="<target_address_as_decimal>"`, and `key="length" value="4096"`. When the ONNX frontend constructs TensorExternalData from this tensor and calls load_external_mem_data(), m_data_location matches ORT_MEM_ADDR (line 117 check passes), m_offset is non-zero (line 121 check passes), and memcpy at line 129 copies 4096 bytes from the attacker-supplied address into the AlignedBuffer. The returned buffer becomes the constant's weight data, leaking memory contents.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON / ASan). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_from_file_is_rejected*'. Expected pre-fix: AddressSanitizer reports an unknown-address / wild read inside std::memcpy (tensor_external_data.cpp:129), or SIGSEGV, instead of a caught ov::Exception. Expected post-fix: the test passes because convert_model throws ov::Exception (error::invalid_external_data) when ORT_MEM_ADDR is supplied via a parsed protobuf. Requires authoring the crafted .onnx fixture noted in the TODOs.

## Suggested fix
The load_external_mem_data() function should never be reachable from an untrusted ONNX file. Add a boolean flag to TensorExternalData (e.g., m_from_trusted_source) set only by the internal/ORT-specific constructor path (line 37), and throw in load_external_mem_data() if the flag is not set. Alternatively, remove the ORT_MEM_ADDR path entirely from the public ONNX model loading flow and gate it behind an explicit allow-list at the call site, never from protobuf-parsed data.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #112.
