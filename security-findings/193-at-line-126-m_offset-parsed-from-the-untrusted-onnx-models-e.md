# Security finding #193: At line 126, `m_offset` (parsed from the untrusted ONNX model's `ex…

**Summary:** At line 126, `m_offset` (parsed from the untrusted ONNX model's `ex…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read of attacker-chosen memory address and length from within the process address space. This can leak secrets (keys, credentials, heap pointers), bypass ASLR, or crash the process by dereferencing an unmapped address. Any application that loads an untrusted ONNX model through the OpenVINO ONNX frontend is affected; the impact is information disclosure and potential DoS.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file → TensorProto::external_data fields ('location', 'offset', 'length') → TensorExternalData constructor (line 19–36) → load_external_mem_data ORT_MEM_ADDR branch

## Description / Root cause
At line 126, `m_offset` (parsed from the untrusted ONNX model's `external_data` map via `std::stoull` at line 24, with no bounds or address-range validation) is unconditionally cast to `char* addr_ptr = reinterpret_cast<char*>(m_offset)`. Line 129 then executes `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)`, reading `m_data_length` bytes from that attacker-supplied address. The only guard (lines 121–124) merely checks that `m_offset != 0 && m_data_length != 0`; it does not verify that `m_offset` falls within any valid, owned memory region. The `ORT_MEM_ADDR` sentinel (`"*/_ORT_MEM_ADDR_/*"`) that gates this branch is itself read from the same untrusted `location` field (line 22), so an attacker-crafted model can activate this branch at will via the `m_tensor_proto` code path in `tensor.hpp:322` and `tensor.cpp:453`.

**Validator analysis:** The vulnType (CWE-822 Untrusted Pointer Dereference) is accurate and the impact (arbitrary in-process read leading to info-disclosure or DoS) is correct. Confirmed by tracing: TensorExternalData(const TensorProto&) populates m_data_location/m_offset/m_data_length purely from the untrusted external_data map (lines 22-26) with no validation; get_ov_constant (tensor.cpp:447-461) and get_external_data (tensor.hpp:317-331) take the load_external_mem_data branch whenever ext_data.data_location()==ORT_MEM_ADDR (lines 455/324), and that string equals the attacker-supplied 'location'. has_external_data() (tensor.hpp:312-313) is satisfied by setting data_location()==EXTERNAL in the proto — all attacker-controllable. load_external_mem_data's only guard (lines 121-124) checks non-zero values, never that m_offset is a valid owned address, then casts and memcpy's (lines 126-129). Unlike load_external_data/load_external_mmap_data, this branch never validates against any file/region size. The ORT_MEM_ADDR sentinel is documented (hpp:85-91) as an in-process-only marker that ORT sets when weights are already in shared memory — i.e. it is meant to be trusted only when the pointer comes from the live m_tensor_place, never from a deserialized file. The proposed fix is correct in spirit and sufficient: gate the ORT_MEM_ADDR branch on m_tensor_place != nullptr in both tensor.hpp:324 and tensor.cpp:455 so a file-origin TensorExternalData can never select the raw-pointer path; throw error::invalid_external_data otherwise. A stronger defense-in-depth is to also stop overloading m_data_location for the sentinel — e.g. add an explicit boolean/origin flag set only by the (location, offset, size) constructor (line 37) — so the marker can never be forged through the proto constructor at all. Either way the load_external_mem_data raw cast should additionally only be reachable from the place-origin constructor.

## Exploit / Proof of Concept
Craft an ONNX model with a tensor initializer whose `data_location` is set to `TensorProto_DataLocation_EXTERNAL`, and whose `external_data` map contains `location="*/_ORT_MEM_ADDR_/*"`, `offset=<target address as decimal string>`, and `length=<number of bytes to read>`. When the model is loaded, `TensorExternalData::TensorExternalData(const TensorProto&)` (line 24) calls `std::stoull` on the offset string and stores it in `m_offset`. `Tensor::get_external_data` / `get_ov_constant` detect `ORT_MEM_ADDR` and call `load_external_mem_data`. Line 126 casts `m_offset` to `char*`; line 129 memcpy's `m_data_length` bytes from that address into an `AlignedBuffer` which becomes an `ov::op::v0::Constant` node — exposing the bytes to the model consumer or crashing if the address is unmapped.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_must_be_rejected*'. Expected pre-fix: ASan SEGV / 'heap-buffer-overflow' or wild read inside ov::frontend::onnx::detail::TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129); expected post-fix: test passes because convert_model throws ov::Exception (error::invalid_external_data) when the ORT_MEM_ADDR location is reached via the model-proto path. TODO: drop the crafted external_data_ort_mem_addr_untrusted.onnx fixture into the onnx frontend models directory + manifest before running.

## Suggested fix
The `ORT_MEM_ADDR` branch must only be taken when the pointer originates from a trusted in-process caller (the `m_tensor_place` path), never from a deserialized ONNX model file. The simplest fix: in both `Tensor::get_external_data` (tensor.hpp:324) and `Tensor::get_ov_constant` (tensor.cpp:455), replace the bare `ext_data.data_location() == detail::ORT_MEM_ADDR` check with a compound guard that also requires `m_tensor_place != nullptr`. If `m_tensor_place` is null, treat `ORT_MEM_ADDR` as an invalid location and throw `error::invalid_external_data`. Additionally, in `load_external_mem_data` itself, assert (or throw) if called when `m_tensor_place`-origin is not guaranteed — or remove the public accessibility of `load_external_mem_data` and only expose it through the trusted-place-only call path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #193.
