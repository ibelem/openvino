# Security finding #417: At line 126, `m_offset` (parsed from the ONNX tensor's `external_da…

**Summary:** At line 126, `m_offset` (parsed from the ONNX tensor's `external_da…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary process-memory read: an attacker who can supply a crafted ONNX model can leak up to `m_data_length` bytes from any readable virtual address in the inference-engine process (weights, heap allocations, stack frames, mapped libraries). If the target address is not mapped, the process crashes (SIGSEGV / access-violation), constituting a reliable DoS. If the address is mapped, sensitive data (keys, weights, application state) is returned as tensor data. Affects any application that loads ONNX models from untrusted sources through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX protobuf model file → TensorExternalData constructor (line 24, std::stoull) → m_offset stored as uint64_t → load_external_mem_data reinterpret_cast to char*

## Description / Root cause
At line 126, `m_offset` (parsed from the ONNX tensor's `external_data["offset"]` string field via `std::stoull` at line 24 with no value constraints) is directly cast to a pointer: `char* addr_ptr = reinterpret_cast<char*>(m_offset)`. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads `m_data_length` bytes from that attacker-controlled virtual address. The only guard (lines 121-124) rejects only the single case where `m_offset == 0 && m_data_length > 0`; any non-zero offset with any non-zero length bypasses it entirely.

**Validator analysis:** The flaw is real and reachable from openvino's own ONNX-file trust boundary. In TensorExternalData(const TensorProto&) (line 19-36) both the location string (line 22) and offset (line 24, std::stoull, no range/validity check) come straight from the attacker-controlled external_data fields of a TensorProto loaded from an .onnx file. get_external_data() (tensor.hpp:317-325) selects the ORT_MEM_ADDR branch purely by string-comparing the attacker-supplied location field against the marker constant, with no flag distinguishing a genuine in-process ORT handoff from a file-loaded proto. load_external_mem_data() (line 116-134) then casts the attacker integer to char* (line 126) and memcpy's m_data_length bytes (line 129); the only guard (121-124) merely rejects offset==0 with length>0. The vulnType (CWE-822 untrusted pointer dereference / CWE-125 OOB read) and the impact (arbitrary process-memory read of up to m_data_length bytes, or SIGSEGV DoS for unmapped addresses) are both accurate. Note the sibling file-path loaders load_external_mmap_data (53-54) and load_external_data (83-84) DO bounds-check offset/length against file_size — this ORT_MEM_ADDR path is the lone unchecked branch, confirming it as an outlier. The proposed fix is correct in direction: the ORT_MEM_ADDR path must not be reachable from a file-backed proto. A robust, minimal fix is to add a constructor flag marking whether the TensorExternalData was built from an untrusted file (TensorExternalData(const TensorProto&)) vs. an in-process ORT TensorPlace handoff (the size_t-pointer ctor at line 37), and have load_external_mem_data() / get_external_data() throw error::invalid_external_data when the ORT_MEM_ADDR path is taken from the file-backed ctor. Validating offset against a registry of pre-registered shared regions (as suggested) is even safer but heavier; gating by construction-origin is sufficient to close the file-load attack surface. I rejected openvinoEp because the cited trust boundary is an .onnx file parsed directly by OpenVINO's ONNX frontend, not the ORT EP handoff, and reachability of an attacker-controlled offset through the EP would require ORT itself to forward an unsanitized marker (out of scope, unconfirmed).

## Exploit / Proof of Concept
Craft an ONNX protobuf with one initializer tensor whose `data_location` field is set to `EXTERNAL` and whose `external_data` repeated field contains three entries: `{key="location", value="*/_ORT_MEM_ADDR_/*"}`, `{key="offset", value="<target_address_decimal>"}`, `{key="length", value="4096"}`. When the ONNX frontend calls `Tensor::get_data()` → `get_external_data()` (tensor.hpp line 322-325), it constructs `TensorExternalData(*m_tensor_proto)`, which stores the decimal-parsed offset in `m_offset` (line 24). `ext_data.data_location() == ORT_MEM_ADDR` is true, so `load_external_mem_data()` is called. The guard at lines 121-124 passes (offset != 0, length != 0). Line 126 casts offset to a `char*`; line 129 copies 4096 bytes from that process address into the returned `AlignedBuffer`, which is then surfaced as the tensor's constant data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// Pre-fix: load_external_mem_data() reinterpret_casts the attacker-supplied
//   external_data["offset"] integer to char* and memcpy's m_data_length bytes
//   -> ASan: heap/unmapped read or SEGV.
// Post-fix: the ORT_MEM_ADDR path is unreachable from a file-backed TensorProto
//   and convert_model() throws ov::Exception (error::invalid_external_data).
//
// This test needs a crafted .onnx fixture and is therefore a SKELETON: the
// model must contain one initializer with data_location=EXTERNAL and
// external_data { key="location" value="*/_ORT_MEM_ADDR_/*" },
// { key="offset" value="<nonzero>" }, { key="length" value="4096" }.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

// TODO: create the crafted fixture below under
//   openvino/src/frontends/onnx/tests/models/ and add it to the test CMake
//   model list. It must set the first initializer's data_location to EXTERNAL
//   and its external_data entries to location="*/_ORT_MEM_ADDR_/*",
//   offset="4096" (any nonzero), length="4096".
OPENVINO_TEST(${FRONTEND_NAME}, onnx_external_ort_mem_addr_from_file_is_rejected) {
    // Pre-fix this does NOT throw and instead dereferences a bogus pointer
    // (ASan/SEGV); post-fix it must throw because the ORT_MEM_ADDR branch is
    // not allowed for file-backed protos.
    EXPECT_THROW(convert_model("crafted_ort_mem_addr.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_ort_mem_addr_from_file_is_rejected*'. Requires a crafted models/crafted_ort_mem_addr.onnx (TODO above). Expected pre-fix under ASan: 'AddressSanitizer: SEGV on unknown address' or 'heap-buffer-overflow READ' inside std::memcpy in TensorExternalData::load_external_mem_data; post-fix the test passes because convert_model throws ov::Exception.

## Suggested fix
Remove or gate the entire `ORT_MEM_ADDR` code path when loading from untrusted ONNX files. This feature was designed for ORT's in-process shared-memory IPC and has no business being reachable from the file-load path. Specifically: (1) In `Tensor::get_external_data()` (tensor.hpp line 324), reject `ORT_MEM_ADDR` if the tensor was created from a file-backed `TensorProto` rather than an in-process ORT handoff — add a boolean flag or separate construction path. (2) In `load_external_mem_data()`, validate that `m_offset` is a pointer within a pre-registered set of allowed shared-memory regions rather than accepting any arbitrary integer. At minimum, replace the current guard with: `if (m_offset == 0 || !is_registered_ort_shared_region(m_offset, m_data_length)) throw error::invalid_external_data{*this};`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #417.
