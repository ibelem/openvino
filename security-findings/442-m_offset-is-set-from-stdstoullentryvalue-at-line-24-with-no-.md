# Security finding #442: `m_offset` is set from `std::stoull(entry.value())` at line 24 with…

**Summary:** `m_offset` is set from `std::stoull(entry.value())` at line 24 with…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who controls both `offset` (= any arbitrary process address, e.g. 0x7ffe00000000) and `length` (= a readable region size) can read arbitrary memory from the inference-engine process into an `AlignedBuffer` that is then returned to the caller and likely embedded in an `ov::op::v0::Constant` node. This is a process-memory information leak: stack data, heap pointers, cryptographic keys, or other sensitive runtime data can be exfiltrated via the model output.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file (attacker-supplied) → raw memory read via reinterpret_cast of model-controlled offset

## Description / Root cause
`m_offset` is set from `std::stoull(entry.value())` at line 24 with no validation. At line 126, it is cast directly to a `char*` via `reinterpret_cast<char*>(m_offset)` and used as the source pointer for `memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)`. The only check is `m_offset != 0` (the truthiness test at line 121), which merely prevents the null address — any other arbitrary address is accepted.

**Validator analysis:** Confirmed real and reachable in openvino. A standalone ONNX model file parsed by the ONNX frontend has m_tensor_place==nullptr, so get_ov_constant (tensor.cpp:449-453) constructs TensorExternalData(*m_tensor_proto), reading location/offset/length straight from the proto's external_data entries (tensor_external_data.cpp:20-30) with std::stoull and no validation. ORT_MEM_ADDR is just the literal string "*/_ORT_MEM_ADDR_/*" (hpp:91) — nothing prevents a model file from setting location to exactly that marker. When it matches (tensor.cpp:455), load_external_mem_data() runs: the only guard is m_offset && m_data_length (line 121), then m_offset is reinterpret_cast to char* (line 126) and memcpy'd m_data_length bytes (line 129) into an AlignedBuffer that becomes a Constant returned to the caller. This is an arbitrary-read process-memory information leak; CWE-822 (Untrusted Pointer Dereference) and the info-leak impact are accurate (it can also crash/SIGSEGV on an unmapped address, a DoS variant). The proposed fix — gating the ORT_MEM_ADDR branch on m_tensor_place != nullptr so only ORT-runtime-injected (in-process) tensors may use the raw-pointer path, while model-file-originated proto tensors are rejected — is correct and sufficient because the marker is only legitimate for ORT in-process sharing, never for file-parsed external_data. Equivalent: in TensorExternalData(*m_tensor_proto) reject location==ORT_MEM_ADDR. openvinoEp is na: the flaw and its fix live wholly in openvino; the EP is merely the legitimate consumer of the in-process mechanism and adds no vulnerable code in plugin_impl.

## Exploit / Proof of Concept
Craft an ONNX model: `location=*/_ORT_MEM_ADDR_/*`, `offset=<target address in decimal, e.g. libc .got.plt address>`, `length=4096`. When processed by OV EP, `load_external_mem_data()` reinterpret-casts `offset` to `char*` at line 126, then `memcpy`s 4096 bytes from that address into an AlignedBuffer. The buffer is returned as a constant tensor whose values can be read via the model output, leaking 4096 bytes of process memory starting at the attacker-chosen address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 untrusted pointer dereference in
// ov::frontend::onnx::detail::TensorExternalData::load_external_mem_data()
// (tensor_external_data.cpp:126-129) reached via Tensor::get_ov_constant()
// (core/tensor.cpp:455) when a *model-file* tensor sets external_data
// location to the ORT_MEM_ADDR marker "*/_ORT_MEM_ADDR_/*" and an arbitrary
// decimal offset. Pre-fix: m_offset is reinterpret_cast<char*> and memcpy'd,
// triggering an arbitrary read (ASan SEGV / heap-buffer-overflow on read or
// silent info-leak). Post-fix: the ORT_MEM_ADDR branch must be rejected for
// proto-originated tensors (m_tensor_place == nullptr), throwing
// ov::Exception (error::invalid_external_data).
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// This needs a crafted .onnx fixture, so it is a SKELETON.

#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"   // import_onnx_model / convert_model helpers

using namespace ov::frontend::onnx::tests;

// TODO: provide models/ort_mem_addr_arbitrary_offset.onnx:
//   a single Constant/initializer whose TensorProto.external_data is:
//     { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//     { key:"offset",   value:"<arbitrary address, e.g. 140187732541440"> }
//     { key:"length",   value:"4096" }
//   and data_location==EXTERNAL so has_external_data() is true.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_ort_mem_addr_from_model_file_rejected) {
    // TODO: confirm the exact fixture-loading macro/helper name in onnx_import.in.cpp
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_offset.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ./ov_onnx_frontend_tests --gtest_filter='*onnx_external_ort_mem_addr_from_model_file_rejected*'. Expected pre-fix under ASan: SEGV / 'AddressSanitizer: SEGV on unknown address' (or heap-buffer-overflow READ) inside std::memcpy in TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129); post-fix the model conversion throws ov::Exception and the test passes. Requires the crafted .onnx fixture noted in the TODO.

## Suggested fix
The `load_external_mem_data()` path is intended for ORT-EP in-process tensor sharing (not model-file-originated tensors). The `TensorExternalData(*m_tensor_proto)` constructor path (tensor.cpp:453) should not permit `ORT_MEM_ADDR` when originating from a model file — add a guard so that `ORT_MEM_ADDR` is only allowed when `m_tensor_place != nullptr` (i.e., the pointer was injected by the ORT runtime, not parsed from the file). In `get_ov_constant` (tensor.cpp:455), check `m_tensor_place != nullptr` before permitting the `ORT_MEM_ADDR` branch.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #442.
