# Security finding #437: At line 121, the only guard before treating `m_offset` as a raw poi…

**Summary:** At line 121, the only guard before treating `m_offset` as a raw poi…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker can read an arbitrary contiguous range of the process's virtual address space (up to 2^64-1 bytes) and have those bytes returned as model weight data. This enables information disclosure of heap contents, stack secrets, mapped file data, cryptographic keys, or other sensitive in-memory data. It can also crash the process if the chosen address is unmapped (DoS). Affected: any application that loads an untrusted ONNX model through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX model file → TensorProto.external_data key-value pairs → TensorExternalData constructor (lines 19-36) → load_external_mem_data()

## Description / Root cause
At line 121, the only guard before treating `m_offset` as a raw pointer is a non-zero check (`bool is_valid_buffer = m_offset && m_data_length`). There is no check that `m_offset` corresponds to any legitimately allocated memory region. At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` blindly converts the attacker-supplied integer to a pointer, and line 129 performs `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` — a read of `m_data_length` bytes from arbitrary address `m_offset`. Both values are parsed from the ONNX model's `external_data` key-value strings via `std::stoull` in the constructor at lines 24 and 26, with no further sanitization.

**Validator analysis:** The vulnType (CWE-125 / arbitrary out-of-bounds read) and impact (arbitrary virtual-address disclosure as weight data, or DoS on unmapped address) are accurate. Data flow is confirmed: a model with data_location=EXTERNAL makes has_external_data() true (tensor.hpp:312-313); get_external_data() (tensor.hpp:317-331) builds TensorExternalData(*m_tensor_proto) which stores the attacker's external_data strings verbatim. If location is the sentinel '*/_ORT_MEM_ADDR_/*' (ORT_MEM_ADDR, hpp:91), tensor.hpp:324 routes to load_external_mem_data(); its only check (cpp:117) is that location==ORT_MEM_ADDR — which the attacker satisfied — and the non-zero check at :121, then memcpy from the raw integer pointer at :129. m_offset/m_data_length come unsanitized from std::stoull at :24/:26. The sentinel was meant to be set ONLY via the in-memory ORT EP constructor (cpp:37-41) where m_offset is a genuine allocator address; nothing stops a disk-parsed model from forging it. The proposed fix (reject m_data_location==ORT_MEM_ADDR inside the TensorProto constructor, cpp:19-30) is correct and sufficient as it severs the only model-file route to load_external_mem_data; the optional m_from_ort_ep private flag is sound defense-in-depth and slightly preferable since it enforces the invariant at the use site (cpp:117) regardless of future call paths.

## Exploit / Proof of Concept
Craft an ONNX protobuf with a tensor initializer whose `data_location` field is `EXTERNAL` and whose `external_data` list contains: `{key:'location', value:'*/_ORT_MEM_ADDR_/*'}`, `{key:'offset', value:'<decimal representation of target virtual address>'}`, `{key:'length', value:'<desired byte count>'}`. When the model is loaded, `has_external_data()` (tensor.hpp:312-313) returns true because `data_location==EXTERNAL`. `get_external_data()` (tensor.hpp:322) then constructs `detail::TensorExternalData(*m_tensor_proto)` which stores the attacker strings as `m_data_location='*/_ORT_MEM_ADDR_/*'`, `m_offset=<target addr>`, `m_data_length=<size>`. Back in `get_external_data()` at tensor.hpp:324-325, `ext_data.data_location() == detail::ORT_MEM_ADDR` is true, so `load_external_mem_data()` is called. Line 121's non-zero check passes because both values are non-zero, and the `memcpy` at line 129 reads `<size>` bytes from `<target addr>` into an `AlignedBuffer` that is then consumed as weight data — leaking the raw memory to the caller.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 arbitrary read in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:117-129.
// A model whose initializer has data_location=EXTERNAL and an external_data
// 'location' entry equal to the ORT_MEM_ADDR sentinel "*/_ORT_MEM_ADDR_/*",
// with attacker-chosen 'offset' (target virtual address) and 'length', must be
// REJECTED at parse time. Pre-fix: load_external_mem_data() memcpy's from the
// forged pointer (ASan: SEGV / heap-buffer-overflow read). Post-fix: the
// TensorProto constructor throws ov::Exception (invalid_external_data).
//
// NOTE: harness is ov_onnx_frontend_tests (style of onnx_import.in.cpp). This
// needs a crafted protobuf fixture, so it is a SKELETON — the .onnx/.prototxt
// asset must be authored by hand because the malicious sentinel cannot be
// emitted by the standard ONNX exporters.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create the fixture model at
//   onnx/models/external_data/ort_mem_addr_forged.prototxt
// A single Constant/initializer tensor with:
//   data_location: EXTERNAL
//   external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
//   external_data { key: "offset"   value: "<any nonzero decimal addr>" }
//   external_data { key: "length"   value: "16" }
// and reference it from a node so get_ov_constant()/get_external_data() runs.
TEST(onnx_external_data, model_cannot_forge_ort_mem_addr_sentinel) {
    // TODO: confirm convert_model() helper name/signature from onnx_import.in.cpp
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_forged.prototxt"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.model_cannot_forge_ort_mem_addr_sentinel . First author the crafted fixture (see TODO). Pre-fix with ASan: AddressSanitizer SEGV / out-of-bounds read in TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129 memcpy from forged pointer). Post-fix: test passes because the TensorProto constructor throws ov::Exception (invalid_external_data) when location==ORT_MEM_ADDR.

## Suggested fix
The `ORT_MEM_ADDR` sentinel must never be accepted from a model file parsed off disk. Add a check at the deserialization point: in `TensorExternalData::TensorExternalData(const TensorProto& tensor)` (line 21-22), after setting `m_data_location = entry.value()`, verify `m_data_location != ORT_MEM_ADDR` and throw `error::invalid_external_data` if it matches. This prevents the attacker from ever setting `m_data_location` to the special marker via the model file. Additionally, `load_external_mem_data()` itself should only be callable on instances constructed via the trusted `TensorExternalData(const std::string& location, size_t offset, size_t size)` constructor (used by the ORT EP when it supplies real allocator addresses); a simple private boolean flag `m_from_ort_ep` (set only by that constructor, false by default) checked at line 117 would enforce this invariant.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #437.
