# Security finding #356: In `TensorExternalData(const TensorProto&)` (line 24), `m_offset` i…

**Summary:** In `TensorExternalData(const TensorProto&)` (line 24), `m_offset` i…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary read of process memory at an attacker-specified address: if the address is mapped (heap, stack, code segment, secrets), `memcpy` copies up to `m_data_length` bytes of that memory into an `AlignedBuffer` that becomes a model constant — creating an information-disclosure primitive (heap/stack secrets, cryptographic keys, etc. can leak through the model's output tensors). If the address is unmapped, the read causes SIGSEGV and crashes the process (denial of service). Affects any application using OpenVINO's ONNX frontend to load attacker-supplied models.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied ONNX model file (.onnx): the TensorProto external_data fields `location`, `offset`, and `length` are parsed from the model without sanitization.

## Description / Root cause
In `TensorExternalData(const TensorProto&)` (line 24), `m_offset` is populated verbatim as `std::stoull(entry.value())` from the proto's `offset` key — an attacker-controlled string. In `load_external_mem_data()`, the only validation (lines 121-124) checks `m_offset != 0 && m_data_length != 0`, which is completely insufficient as an address legitimacy check. Line 126 then performs `char* addr_ptr = reinterpret_cast<char*>(m_offset)`, treating the raw integer as a process-memory pointer, and line 129 does `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` — reading `m_data_length` bytes from that attacker-chosen address. There is no check that `m_offset` points to a legitimately allocated buffer; it is used as a raw address. In `tensor.cpp` line 453, the `TensorExternalData(*m_tensor_proto)` constructor path is taken whenever `m_tensor_place == nullptr` (standard file-based ONNX loading), so this is fully reachable from an untrusted model file.

**Validator analysis:** Confirmed real and reachable in openvino. For a file-based model load, m_tensor_place is null so TensorExternalData(*m_tensor_proto) is constructed (tensor.cpp:453), parsing attacker-controlled `location`, `offset`, `length` from the TensorProto external_data (ctor lines 19-36). An attacker sets location to the ORT_MEM_ADDR marker `*/_ORT_MEM_ADDR_/*` (defined hpp:91), which routes to load_external_mem_data (tensor.cpp:455). The only guard there is `m_offset && m_data_length` (line 121) — no validation that m_offset is a legitimately mapped buffer. Line 126 reinterprets the integer as a pointer and line 129 memcpy's m_data_length bytes into an AlignedBuffer that becomes a model constant. The element_count vs shape check at tensor.cpp:467 runs only AFTER the memcpy, so it cannot prevent the OOB/arbitrary read or SIGSEGV. CWE-822 (untrusted pointer dereference) and CWE-125 are accurate; impact (info disclosure / DoS) is correct. The proposed fix is correct and sufficient: gating ORT_MEM_ADDR behind a m_from_proto flag (only allow it when constructed via the trusted (location,offset,size) ctor used at tensor.cpp:450-452 with m_tensor_place!=nullptr) cleanly removes the file-driven attacker path; the data_length upper-bound is reasonable defense-in-depth. A simpler equivalent: never honor ORT_MEM_ADDR for proto-constructed instances. unitTest is a skeleton because triggering requires a crafted .onnx fixture with an external_data ORT_MEM_ADDR marker, which cannot be authored inline safely (arbitrary-address memcpy).

## Exploit / Proof of Concept
1. Craft an ONNX model with a tensor whose `external_data` entries are: `location = '*/_ORT_MEM_ADDR_/*'`, `offset = <target address as decimal string, e.g. '140737488355328'>`, `length = '4096'`. 2. Feed this model to OpenVINO's ONNX frontend (e.g., `ov::Core::read_model`). 3. `m_tensor_place` is null for file-based loading, so `TensorExternalData(*m_tensor_proto)` is called (tensor.cpp:453), storing the attacker's integer as `m_offset`. 4. `data_location() == ORT_MEM_ADDR` (tensor.cpp:455) triggers `load_external_mem_data()`. 5. The check `m_offset && m_data_length` (line 121) passes since both are non-zero. 6. `reinterpret_cast<char*>(m_offset)` at line 126 materialises the arbitrary pointer; `memcpy` at line 129 reads 4096 bytes from that address into the AlignedBuffer. 7. The buffer is returned as the tensor's constant value, accessible to the caller. No exception or guard is encountered on this path.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data). Pre-fix: a file-parsed TensorProto whose
// external_data location == ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") with attacker offset is
// reinterpret_cast to a pointer and memcpy'd from -> arbitrary read / SIGSEGV (ASan: SEGV
// or invalid read). Post-fix: the proto-constructed ORT_MEM_ADDR path is rejected with
// ov::Exception (error::invalid_external_data), so convert_model must throw and never deref.
//
// Place in openvino/src/frontends/onnx/tests/onnx_import.in.cpp style; target ov_onnx_frontend_tests.

#include "onnx_utils.hpp"            // TODO: confirm helper header name in the tests/ dir
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add a crafted fixture model file under tests/models/, e.g.
//   external_data_ort_mem_addr.onnx
// containing a single initializer with TensorProto.external_data entries:
//   key="location" value="*/_ORT_MEM_ADDR_/*"
//   key="offset"   value="140737488355328"   // bogus/attacker address
//   key="length"   value="4096"
// (cannot be expressed inline as prototxt safely; must be a binary .onnx fixture)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_is_rejected) {
    // After the fix, the ORT_MEM_ADDR path constructed from a parsed proto (m_tensor_place
    // == nullptr, m_from_proto == true) must be rejected rather than dereferencing m_offset.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Add fixture external_data_ort_mem_addr.onnx under src/frontends/onnx/tests/models/. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_from_file_is_rejected*'. Pre-fix expected: ASan AddressSanitizer SEGV / 'invalid read of size 4096' (or unbounded read) inside TensorExternalData::load_external_mem_data at tensor_external_data.cpp:129. Post-fix expected: test passes (ov::Exception thrown, no memcpy from attacker address).

## Suggested fix
The `ORT_MEM_ADDR` mechanism was designed for in-process ORT shared-memory tensors, not for use with untrusted model files. The fix has two parts: (1) Block this path when the `TensorExternalData` was constructed from a raw `TensorProto` (i.e., from a file). Add a boolean `m_from_proto` flag set in the proto constructor, and in `load_external_mem_data()` reject the call (`throw error::invalid_external_data`) when `m_from_proto` is true — only allow the ORT-memory path when constructed via `TensorExternalData(const std::string&, size_t, size_t)` from a trusted in-process caller (tensor.cpp line 450-452 with `m_tensor_place != nullptr`). (2) As defense-in-depth, validate `m_data_length` has a sane upper bound before the `AlignedBuffer` allocation (line 127) to prevent allocation-size DoS.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #356.
