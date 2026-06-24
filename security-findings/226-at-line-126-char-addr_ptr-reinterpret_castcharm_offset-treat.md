# Security finding #226: At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` t…

**Summary:** At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` t…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary in-process memory read: any address and length encodable as uint64 can be read. The copied bytes are returned as tensor constant data, which the caller can then inspect (e.g. through the resulting ov::Constant or ov::AlignedBuffer). A malicious ONNX file loaded by any application using the OpenVINO ONNX frontend can exfiltrate heap/stack/code contents, including secrets such as private keys, passwords, or model weights from other loaded models (CWE-200 secondary). On a 64-bit system the full 64-bit address space is reachable; unmapped addresses will cause a SIGSEGV/access violation, which is also a reliable crash/DoS.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX model file → OpenVINO ONNX frontend (external_data entries parsed in TensorExternalData constructor at line 24)

## Description / Root cause
At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` treats the 64-bit integer `m_offset` — parsed directly from the untrusted ONNX protobuf field `external_data["offset"]` via `std::stoull` at line 24 — as a raw process memory address. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads `m_data_length` bytes from that attacker-controlled address. The only guard (lines 121-125) throws only when `m_offset == 0 && m_data_length != 0`; any nonzero offset with nonzero length reaches the memcpy unconditionally. There is no check that `m_data_location == ORT_MEM_ADDR` is reached only from trusted (ORT-internal) sessions: the call path `Tensor::get_external_data<T>()` in core/tensor.hpp:322 constructs `TensorExternalData(*m_tensor_proto)` directly from a file-parsed TensorProto when `m_tensor_place == nullptr`, then dispatches to `load_external_mem_data()` when `data_location() == ORT_MEM_ADDR` (line 324-325), so an attacker can craft this path entirely from a .onnx file.

**Validator analysis:** CWE-125/arbitrary in-process read is accurate and the impact (info-disclosure of process memory / SIGSEGV DoS) is correct. Confirmed: TensorExternalData(const TensorProto&) at line 19 reads location, offset, length directly from the untrusted external_data map (lines 22-26). In get_external_data() (tensor.hpp:316-332), when m_tensor_place==nullptr (the normal .onnx file-parse path), the proto-based constructor is used (line 322) and, if data_location()==ORT_MEM_ADDR (an attacker-supplied string, line 324), load_external_mem_data() is invoked. load_external_mem_data() only guards offset==0&&length!=0 (lines 121-125); any nonzero offset with nonzero length reaches reinterpret_cast<char*>(m_offset) (126) and memcpy of attacker-chosen length from an attacker-chosen address (129). The proposed fix is correct and sufficient in principle: the ORT_MEM_ADDR/in-memory-address path must be reachable ONLY when constructed from the trusted ORT m_tensor_place source. Cleanest implementation is the finding's layer (2): in tensor.hpp:324, reject data_location()==ORT_MEM_ADDR whenever m_tensor_place==nullptr (file-parsed proto), throwing before load_external_mem_data(). Layer (1)'s m_from_ort_session flag is a good defense-in-depth so the dangerous primitive cannot be invoked from the public proto constructor at all. For openvinoEp the same code is present but the attacker cannot supply the in-memory address (ORT sets it), so it is rejected there.

## Exploit / Proof of Concept
Craft an ONNX protobuf with a TensorProto whose `data_location = EXTERNAL` and `external_data` map containing: `{"location": "*/_ORT_MEM_ADDR_/*", "offset": "<target_address_as_decimal>", "length": "4096"}`. Load this model through `ov::Core::read_model()`. The ONNX frontend constructs `TensorExternalData` from the protobuf (constructor lines 19-36), parses `m_offset = stoull("<target_address>")` and `m_data_length = 4096`. When the tensor's constant value is materialized, `get_external_data<T>()` (tensor.hpp:322) creates a `TensorExternalData` from the raw proto, checks `data_location() == ORT_MEM_ADDR` (passes — attacker set this string), calls `load_external_mem_data()`. Lines 121-123 pass because both values are nonzero. Line 129 executes `memcpy(aligned_memory, <target_address>, 4096)`, reading 4096 bytes of process memory starting at the attacker-specified address into the output buffer, which is returned as the tensor's data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-125 OOB/arbitrary read at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// A file-parsed ONNX TensorProto with external_data location == "*/_ORT_MEM_ADDR_/*"
// and a nonzero offset/length must NOT be allowed to reach load_external_mem_data()
// (which memcpy's from reinterpret_cast<char*>(offset)). After the fix, importing
// such a model must throw instead of dereferencing an attacker-chosen address.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
//
// NOTE: This needs a crafted .onnx fixture (initializer with data_location=EXTERNAL,
//       external_data{location:"*/_ORT_MEM_ADDR_/*", offset:"<addr>", length:"4096"}).
//       Pre-fix this either reads arbitrary memory or SIGSEGVs (ASan-detectable);
//       post-fix convert_model throws. Marked skeleton because the binary fixture
//       and exact convert_model helper must be supplied/verified against the tree.

#include "onnx_utils.hpp"   // TODO: confirm helper header providing convert_model()
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add models/external_data_ort_mem_addr.onnx fixture encoding:
//   initializer { data_location: EXTERNAL,
//                 external_data: [ {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                                  {key:"offset",   value:"<some nonzero decimal>"},
//                                  {key:"length",   value:"4096"} ] }
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected) {
    // Must throw: file-parsed proto must never use an in-process memory address.
    EXPECT_THROW(convert_model("external_data/external_data_ort_mem_addr.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_rejected*. Pre-fix (with ASan): heap-buffer-overflow / SEGV in TensorExternalData::load_external_mem_data memcpy (tensor_external_data.cpp:129) reading from reinterpret_cast<char*>(m_offset), or arbitrary read with no throw -> test FAILS. Post-fix: convert_model throws ov::Exception before load_external_mem_data() (guard added at tensor.hpp:324 for m_tensor_place==nullptr) -> test PASSES. TODO: author the .onnx fixture and confirm the convert_model helper name in the onnx frontend test tree.

## Suggested fix
The `ORT_MEM_ADDR` path must only be reachable when the TensorExternalData is constructed from an ORT-internal, trusted source — not from a file-parsed TensorProto. Fix this in two layers: (1) In `TensorExternalData::load_external_mem_data()`, add an explicit runtime guard that rejects all calls not originating from the trusted second constructor (`TensorExternalData(const std::string& location, size_t offset, size_t size)`). The simplest approach is to add a `bool m_from_ort_session = false` flag set only by that constructor, and check `OPENVINO_ASSERT(m_from_ort_session)` at the top of `load_external_mem_data()`. (2) In `Tensor::get_external_data<T>()` (tensor.hpp:324), when `m_tensor_proto` path is taken (i.e. `m_tensor_place == nullptr`), explicitly reject `data_location() == ORT_MEM_ADDR` with an exception before calling `load_external_mem_data()`, since file-parsed tensors must never be allowed to use in-process memory addresses.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #226.
