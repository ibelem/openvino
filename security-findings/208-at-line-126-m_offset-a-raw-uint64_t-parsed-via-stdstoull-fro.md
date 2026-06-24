# Security finding #208: At line 126, `m_offset` (a raw `uint64_t` parsed via `std::stoull` …

**Summary:** At line 126, `m_offset` (a raw `uint64_t` parsed via `std::stoull` …

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary process memory read: an attacker who can supply a crafted `.onnx` file can exfiltrate any readable memory in the OpenVINO host process (stack, heap, mapped libraries, secrets, keys). On 64-bit systems the full 64-bit address space is addressable. Additionally, an unmapped or inaccessible address causes a segfault (crash/DoS). Affects any application calling ov::Core::read_model() or the ONNX frontend on an attacker-controlled model file.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file (protobuf) → TensorProto::external_data entries → m_data_location, m_offset, m_data_length set in constructor lines 22-26

## Description / Root cause
At line 126, `m_offset` (a raw `uint64_t` parsed via `std::stoull` from the protobuf `offset` key at constructor line 24) is cast directly to `char*` via `reinterpret_cast<char*>(m_offset)`. At line 129, `memcpy` reads `m_data_length` bytes (also attacker-controlled, constructor line 26) from that address. The only guard at line 117 checks `m_data_location != ORT_MEM_ADDR` — this check **permits** the ORT_MEM_ADDR path rather than restricting it, and `m_data_location` is itself set from the untrusted protobuf at line 22. There is no check that the ONNX model originated from a trusted in-process caller (the ORT shared-memory mechanism) vs. a file on disk.

**Validator analysis:** Confirmed real and reachable in openvino. has_external_data() (tensor.hpp:312-313) returns true for data_location==EXTERNAL; get_external_data() builds TensorExternalData from *m_tensor_proto when m_tensor_place is null (the ov::Core::read_model file path), and the protobuf-parsing ctor (lines 19-36) copies the 'location' string verbatim with no restriction, so it can equal ORT_MEM_ADDR ('*/_ORT_MEM_ADDR_/*'). Line 324 then dispatches to load_external_mem_data(), whose only guard (line 117) merely PERMITS the ORT_MEM_ADDR path; line 121 only requires offset&&length nonzero. m_offset is stoull-parsed from an attacker string and cast straight to char* (line 126), then memcpy'd (line 129) → arbitrary process-memory read / crash. The vuln type CWE-822/CWE-125 and the arbitrary-read+DoS impact are accurate. The marker is an ORT in-process IPC feature that must never be settable from a file-loaded model. The proposed fix is correct and sufficient: rejecting m_data_location==ORT_MEM_ADDR inside the TensorProto ctor (the file-parse path) closes the file boundary while leaving the trusted 3-arg ctor (used by EP/m_tensor_place) intact; this is cleaner and lower-risk than range-checking offsets. Recommend the ctor-level guard as primary fix.

## Exploit / Proof of Concept
Craft an ONNX protobuf with one initializer tensor whose `data_location` enum is set to `EXTERNAL` and whose `external_data` map contains: `{location: "*/_ORT_MEM_ADDR_/*", offset: "<target_address_as_decimal>", length: "<bytes_to_read>"}`. When the frontend parses this tensor, `TensorExternalData` constructor (lines 22, 24, 26) stores the attacker values verbatim. `get_external_data()` in tensor.hpp line 324 sees `ORT_MEM_ADDR` and calls `load_external_mem_data()`. Line 117's guard passes (location IS ORT_MEM_ADDR). Line 121 `is_valid_buffer` passes if both offset and length are non-zero. Line 126 casts the decimal-encoded address to `char*`. Line 129 `memcpy` copies `length` bytes from that address into an `AlignedBuffer`, which is then returned to the caller — enabling the attacker to recover the copied bytes through normal model output.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_rejected*'. Requires a crafted fixture onnx/tests/models/external_data/ort_mem_addr_offset.onnx (data_location=EXTERNAL, external_data location='*/_ORT_MEM_ADDR_/*', offset set to an arbitrary address, length=64). Pre-fix: ASan reports SEGV / heap-buffer-overflow read from the reinterpret_cast<char*>(m_offset) memcpy at tensor_external_data.cpp:129 (or a hard segfault). Post-fix: convert_model throws ov::Exception (error::invalid_external_data) and the EXPECT_THROW passes.

## Suggested fix
The ORT_MEM_ADDR mechanism is an ORT-internal IPC feature that must never be reachable from file-loaded models. Fix on two levels: (1) In `load_external_mem_data()`, add a strict allow-list or process-address-space range check — at minimum, validate that `m_offset` is within a known registered shared-memory region (pass a set of valid `{base, length}` pairs from the trusted caller). (2) At the ONNX frontend load boundary, reject any model loaded from a file (i.e., when model_dir is a real filesystem path) that contains `location == ORT_MEM_ADDR` — e.g., in `TensorExternalData::TensorExternalData(const TensorProto&)` after line 22, throw `error::invalid_external_data` if `m_data_location == ORT_MEM_ADDR`. Example: `if (m_data_location == ORT_MEM_ADDR) { throw error::invalid_external_data{"ORT_MEM_ADDR location is not permitted in file-loaded models"}; }` This ensures the only path that can set ORT_MEM_ADDR is the trusted two-argument constructor `TensorExternalData(location, offset, size)` called by trusted in-process code.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #208.
