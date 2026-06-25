# Security finding #381: At line 121, the only guard before converting `m_offset` to a point…

**Summary:** At line 121, the only guard before converting `m_offset` to a point…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read from any address within the process's virtual address space. An attacker who can supply a crafted ONNX model with `location = "*/_ORT_MEM_ADDR_/*"`, a non-zero `offset` (the target address), and a non-zero `length` can exfiltrate process memory (heap, stack, mapped libraries, secrets/keys), bypass ASLR, or crash the inference engine with SIGSEGV if the address is unmapped. The read data is copied into a SharedBuffer that may be returned to the caller, enabling information disclosure.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model's `external_data.offset` string field (parsed at constructor line 24 via `std::stoull`) → `uint64_t m_offset` → `reinterpret_cast<char*>(m_offset)` used as memcpy source

## Description / Root cause
At line 121, the only guard before converting `m_offset` to a pointer is `bool is_valid_buffer = m_offset && m_data_length` — a mere non-zero check. Any non-zero attacker-supplied integer trivially passes. At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` converts the raw attacker-controlled integer directly into a pointer, which is then immediately used as the source in `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` at line 129. No address-range validation, no alignment check, no bounds against any known-valid buffer is performed.

**Validator analysis:** Confirmed real and reachable from openvino's ONNX-frontend trust boundary. In the file-load path m_tensor_place==nullptr (Tensor ctor tensor.hpp:191 sets it null), so get_external_data() (tensor.hpp:316-332) builds TensorExternalData(*m_tensor_proto), whose ctor (tensor_external_data.cpp:19-36) reads location/offset/length straight from the protobuf external_data entries. Nothing prevents a crafted model from setting location="*/_ORT_MEM_ADDR_/*"; tensor.hpp:324 then calls load_external_mem_data(), whose only check (tensor_external_data.cpp:121) is `m_offset && m_data_length` (non-zero), after which line 126 does reinterpret_cast<char*>(m_offset) and line 129 memcpy's m_data_length bytes FROM that attacker address. This is a genuine CWE-822 untrusted-pointer-dereference giving arbitrary in-process read (or SIGSEGV) into a returned SharedBuffer — vuln type and impact are accurate. The proposed fix (range whitelist) is a workable mitigation but fragile; the cleaner and sufficient fix is to honor the ORT_MEM_ADDR/in-memory-pointer path ONLY when TensorExternalData was built through the trusted size_t-offset constructor (tensor_external_data.cpp:37 / tensor.hpp:319-321, where offset = reinterpret_cast<size_t>(m_tensor_place->get_data())) and to REJECT location==ORT_MEM_ADDR whenever it originates from the protobuf-parsing constructor (line 19). A boolean `m_from_trusted_memptr` flag set only by the size_t ctor, checked at the top of load_external_mem_data(), removes the attack surface entirely. The EP repo is na: the defect is not in EP code.

## Exploit / Proof of Concept
1. Craft an ONNX protobuf where one tensor's `external_data` list has three entries: `{key="location", value="*/_ORT_MEM_ADDR_/*"}`, `{key="offset", value="<target_address_as_decimal>"}`, `{key="length", value="4096"}`. 2. Feed the model to OpenVINO's ONNX frontend. 3. `TensorExternalData` constructor (line 24) calls `std::stoull` on the offset string and stores the integer in `m_offset`. 4. When `load_external_mem_data()` is called, line 121 computes `is_valid_buffer = (non-zero) && (4096)` → `true`, bypassing the only guard. 5. Line 126 casts `m_offset` to `char*`. 6. Line 129 memcpy reads 4096 bytes from the attacker-specified address into `aligned_memory`, which is then returned as the tensor buffer — leaking 4096 bytes of process memory to the caller.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-129
// (TensorExternalData::load_external_mem_data). A crafted ONNX model whose constant
// tensor declares data_location=EXTERNAL with external_data
//   location = "*/_ORT_MEM_ADDR_/*", offset = <attacker addr>, length = <N>
// reaches load_external_mem_data() via core/tensor.hpp:324 and performs
//   memcpy(dst, reinterpret_cast<char*>(offset), N) — an arbitrary in-process read.
// PRE-FIX: the model is accepted and the cast/memcpy runs on the attacker offset
//          (ASan: SEGV / heap-buffer-overflow on read, or silent info-leak).
// POST-FIX: the frontend must reject an ORT_MEM_ADDR location that came from the
//           parsed protobuf (not from the trusted in-memory ctor), so convert_model throws.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// NOTE: needs a crafted .onnx fixture; symbol names below mirror the existing suite
// but the fixture path/loader must be confirmed against the local test tree.
#include "onnx_utils.hpp"   // TODO: confirm helper header providing convert_model()
#include "common_test_utils/test_case.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add models/ort_mem_addr_arbitrary_read.onnx — a model with one Constant/initializer
//       tensor: data_location=EXTERNAL, external_data=[
//         {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//         {key:"offset",   value:"<nonzero decimal, e.g. a bogus address>"},
//         {key:"length",   value:"4096"}]
TEST(onnx_importer, ort_mem_addr_offset_from_model_is_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='onnx_importer.ort_mem_addr_offset_from_model_is_rejected'. Provide the crafted fixture models/ort_mem_addr_arbitrary_read.onnx (Constant tensor, data_location=EXTERNAL, external_data location="*/_ORT_MEM_ADDR_/*", offset=nonzero, length=4096). Pre-fix the test FAILS: convert_model does not throw and AddressSanitizer reports SEGV/heap-buffer-overflow on the read at tensor_external_data.cpp:129 (memcpy reading reinterpret_cast<char*>(m_offset)). Post-fix the frontend rejects the model-supplied ORT_MEM_ADDR location and convert_model throws ov::Exception, so the test PASSES.

## Suggested fix
The `ORT_MEM_ADDR` feature implies the offset value is a real in-process pointer placed there by ORT's own runtime, not a value from an untrusted external model file. Two complementary fixes are needed: (1) Ensure `load_external_mem_data()` is only callable from a trusted internal path (i.e., never invoked when parsing an external, untrusted ONNX file). (2) If the function must remain, validate that `m_offset` falls within a whitelist of known-valid AlignedBuffer/SharedBuffer address ranges rather than accepting any non-zero integer. At minimum, add a compile-time or runtime assertion that the data location and address were supplied by trusted ORT internals, not parsed from the model bytes. Example guard at line 122: `if (m_offset < min_valid_addr || m_offset + m_data_length > max_valid_addr) throw error::invalid_external_data{*this};` where the valid range is tracked by the session manager.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #381.
