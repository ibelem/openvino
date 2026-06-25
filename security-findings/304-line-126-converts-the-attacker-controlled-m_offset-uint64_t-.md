# Security finding #304: Line 126 converts the attacker-controlled `m_offset` (uint64_t, par…

**Summary:** Line 126 converts the attacker-controlled `m_offset` (uint64_t, par…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read of process memory: an attacker who supplies a crafted ONNX model can specify any aligned non-zero address as `offset` and any non-zero size as `length` (both parsed via std::stoull with no range restriction). The resulting AlignedBuffer is returned as tensor data to the caller and — depending on how the model output is surfaced — the raw process memory can be exfiltrated (info-leak). On addresses that are not mapped, `memcpy` triggers a segmentation fault (DoS). Because model loading typically occurs on a trusted backend with access to weight data, crypto keys, or other secrets, the information-leakage impact is high.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file → TensorExternalData constructor (line 26 `std::stoull(entry.value())`) → load_external_mem_data

## Description / Root cause
Line 126 converts the attacker-controlled `m_offset` (uint64_t, parsed from the ONNX model's `offset` field) directly into a `char*` pointer via `reinterpret_cast<char*>(m_offset)`. Line 129 then passes this pointer as the source of `std::memcpy` of `m_data_length` bytes into an AlignedBuffer. The only guard (lines 121-124) only checks that `m_offset != 0 && m_data_length != 0`; it performs no validation that the derived pointer is legitimate, falls within any safe range, or belongs to actual external-data memory.

**Validator analysis:** The flaw is real and reachable from openvino's own ONNX-frontend trust boundary (convert_model on a crafted .onnx). Tensor::get_external_data() builds a TensorExternalData directly from the untrusted TensorProto (tensor.hpp:322), and the ORT_MEM_ADDR branch is checked FIRST (tensor.hpp:324) regardless of mmap_cache, so a model that sets external_data location='*/_ORT_MEM_ADDR_/*' with an arbitrary 'offset' and 'length' will reach load_external_mem_data(). There m_offset (parsed via std::stoull at line 24) is reinterpret_cast to char* (line 126) and used as the memcpy source (line 129); the only guard (lines 121-124) merely checks both values are non-zero. This is an attacker-controlled untrusted pointer dereference → arbitrary process-memory read (info leak) or SIGSEGV (DoS). CWE-822 and the impact are accurate; the mmap/file paths (lines 53-54, 83-84) bound offset/length against file_size, but the mem path has no analogous registry check. The proposed fix is conceptually correct: the ORT_MEM_ADDR path must be gated to the programmatic/trusted constructor only. Best concrete fix: add a bool m_from_trusted_ptr set true only in the (location,offset,size) ctor (tensor.hpp:37-41 path / tensor.hpp:319-321) and false in the TensorProto ctor, then in load_external_mem_data() throw invalid_external_data unless m_from_trusted_ptr is true — i.e., refuse ORT_MEM_ADDR whenever it originated from a parsed model file. openvinoEp is na because the defective code is entirely in openvino and the EP path supplies a vetted pointer through the trusted ctor.

## Exploit / Proof of Concept
Craft an ONNX model with an external-data entry whose `location` field is `*/_ORT_MEM_ADDR_/*` (the ORT_MEM_ADDR sentinel), `offset` set to any target address in the process (e.g., `0x7fff00000000` on a 64-bit Linux process to probe stack memory), and `length` set to e.g. `4096`. OpenVINO's ONNX frontend parses the values with `std::stoull`, stores them in `m_offset`/`m_data_length`, detects the ORT_MEM_ADDR sentinel, passes the `is_valid_buffer` check (both non-zero), casts `m_offset` to `char*`, and `memcpy`s 4096 bytes from that address into the returned SharedBuffer whose contents are passed back to the caller as tensor weights.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_from_file_rejected*'. Requires adding the crafted models/external_data/ort_mem_addr_offset.onnx fixture (TODO). Pre-fix expectation under ASan: heap/segv 'SEGV on unknown address' or invalid read inside load_external_mem_data() (tensor_external_data.cpp:129 memcpy); post-fix: test passes because convert_model throws ov::Exception.

## Suggested fix
The `load_external_mem_data` path must not accept a raw address from the model file. Either (a) restrict this path to internal/ORT IPC usage only: validate at the call-site that `m_offset` was set programmatically (not loaded from a disk model) and refuse to call `load_external_mem_data` for any model loaded from untrusted file input; or (b) replace the raw pointer cast with a safer mechanism: accept a pre-vetted `void*` pointer through a separate trusted API rather than repurposing the `offset` field as an address. At minimum, add an explicit check that the derived address+length range lies within a registered shared-memory region before the `memcpy`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #304.
