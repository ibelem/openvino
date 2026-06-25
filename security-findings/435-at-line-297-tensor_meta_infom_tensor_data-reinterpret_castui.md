# Security finding #435: At line 297, `tensor_meta_info.m_tensor_data = reinterpret_cast<uin…

**Summary:** At line 297, `tensor_meta_info.m_tensor_data = reinterpret_cast<uin…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Any downstream consumer of `TensorMetaInfo::m_tensor_data` (e.g., `DecoderProtoTensor::get_tensor_info()`, constant folding, tensor copy) will dereference an arbitrary attacker-controlled address, yielding an arbitrary-address read (CWE-125). On systems with no guard pages at the target address this is an information disclosure; in most configurations it causes an immediate access violation / crash (DoS). If the attacker also controls `m_tensor_data_size`, a read of up to 2^64-1 bytes can be requested, amplifying the impact.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX model bytes → ORT OV EP → GraphIteratorProto::initialize(std::shared_ptr<ModelProto>) → extract_tensor_external_data

## Description / Root cause
At line 297, `tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset)` blindly casts the attacker-supplied integer (from `std::stoull(entry.value())` at line 287) into a raw pointer. When `ext_location == detail::ORT_MEM_ADDR` (the string `"*/_ORT_MEM_ADDR_/*"`), there is no null check, no address-range validation, no alignment check, and no origin check on `ext_data_offset` before it is stored as `m_tensor_data`. `m_tensor_data_size` (line 298) is also set directly from the attacker-supplied `ext_data_length` (line 289). `initialize(std::shared_ptr<ModelProto>)` at line 531 accepts the ModelProto verbatim with no stripping or validation of `data_location` or `external_data` entries.

**Validator analysis:** The defect is real: at graph_iterator_proto.cpp:294-300 the ORT_MEM_ADDR branch fires purely on the string value read from the serialized protobuf and then does reinterpret_cast<uint8_t*>(ext_data_offset) with m_tensor_data_size taken verbatim from the model — no provenance, null, range, or alignment check. The same unchecked cast appears at :364 (Internal_MMAP/Stream). CWE-822 (untrusted pointer dereference) leading to arbitrary-address read (CWE-125) / DoS is an accurate characterization for the file-load boundary, because nothing prevents an attacker .onnx from carrying location='*/_ORT_MEM_ADDR_/*' with an arbitrary offset, and extract_tensor_external_data checks the marker BEFORE any memory-mode gating. For the EP repo, however, the threat model collapses: ORT_MEM_ADDR is by design (header comment :85-91) an ORT-internal marker whose offset is an address ORT itself assigns to its own in-memory weight buffers when serializing a subgraph for the EP; an attacker editing the model graph cannot make ORT emit an arbitrary address, so the defect is not reachable from the EP trust boundary as stated — hence openvinoEp rejected, openvino validated. The proposed fix is only partially correct: the option (c) heuristic guard (offset>0x10000, alignment) is insufficient — a 64-bit aligned arbitrary address trivially passes it and still yields an OOB read. The authoritative fix is option (a)/(b): carry an explicit provenance flag on GraphIteratorProto so the ORT_MEM_ADDR branch is honored ONLY when the iterator was constructed via the trusted ORT in-process path (initialize(std::shared_ptr<ModelProto>) set by the EP), and is rejected (throw) for any model loaded from a file/untrusted bytes via initialize(path).

## Exploit / Proof of Concept
1) Craft an ONNX protobuf where an initializer tensor has `data_location=EXTERNAL`, and `external_data` key-value pairs: `{"location": "*/_ORT_MEM_ADDR_/*", "offset": "<target uint64>", "length": "8"}`. 2) Feed these bytes to the OV EP; ORT parses them into a `ModelProto` and calls `GraphIteratorProto::initialize(model)` (line 531) — no sanitisation occurs. 3) During graph iteration, `extract_tensor_meta_info` detects `data_location == EXTERNAL` (line 394-396) and calls `extract_tensor_external_data`. 4) The loop at lines 283-292 sets `ext_data_offset = stoull("<target uint64>")` and `ext_location = "*/_ORT_MEM_ADDR_/*"`. 5) The branch at line 294 fires, storing the attacker address into `m_tensor_data` (line 297). 6) Subsequent tensor-data access (e.g., constant-folding pipeline) reads `m_tensor_data[0..7]` — an arbitrary memory read from the attacker-chosen address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-822 at
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-300
// where extract_tensor_external_data() blindly casts the protobuf-supplied
// 'offset' integer to uint8_t* (m_tensor_data) when external_data.location ==
// "*/_ORT_MEM_ADDR_/*". A model loaded from a FILE must never be allowed to set
// an ORT_MEM_ADDR pointer; the fixed frontend must reject it (throw) instead of
// dereferencing an attacker-chosen address.
//
// Pre-fix: convert_model() succeeds (or later derefs the bogus pointer -> ASan
//          SEGV / heap-buffer-overflow during constant folding).
// Post-fix: convert_model() throws ov::Exception because ORT_MEM_ADDR is
//          disallowed on the file-load trust boundary.
//
// TODO: this test needs a crafted binary fixture
//   models/ort_mem_addr_external_data.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data = { location:"*/_ORT_MEM_ADDR_/*", offset:"4096", length:"8" }
// Generate it with a small protobuf script mirroring the ONNX TensorProto schema
// (cannot be produced as plain text here).

#include "onnx_utils.hpp"   // TODO: confirm the FrontEndTestUtils helper header used by onnx_import.in.cpp
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TEST mirrors the style of onnx_import.in.cpp (convert_model + EXPECT_THROW).
TEST(onnx_import_external_data, ort_mem_addr_pointer_rejected_on_file_load) {
    // TODO: place crafted fixture under the frontend test models dir and
    //       reference it the same way other onnx_import.in.cpp tests do.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_external_data.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_import_external_data.ort_mem_addr_pointer_rejected_on_file_load'. Pre-fix expectation: ASan reports SEGV / 'unknown-crash' on arbitrary address (e.g. 0x1000) during tensor-data access, or the EXPECT_THROW fails because no exception is raised. Post-fix expectation: convert_model throws ov::Exception (ORT_MEM_ADDR disallowed for file-loaded models) and the test passes cleanly. TODO: provide the crafted external_data/ort_mem_addr_external_data.onnx fixture.

## Suggested fix
In the `ORT_MEM_ADDR` branch, verify the pointer comes from a trusted ORT-internal provenance rather than an external model file. One approach: (a) reject `ORT_MEM_ADDR` entirely when the model was loaded from a file or untrusted bytes (i.e., when the graph iterator was not constructed via the ORT in-process shared-memory path); (b) if the path must be supported, require that `ext_data_offset` and `ext_data_length` were set by ORT itself (e.g., by accepting them through a separate validated API rather than reading them from the serialised protobuf); (c) at minimum add a runtime sanity guard: `OPENVINO_ASSERT(ext_data_offset != 0 && ext_data_offset > 0x10000ULL && ext_data_length > 0, "ORT_MEM_ADDR: suspicious pointer or zero length");` plus an alignment check (`ext_data_offset % alignof(std::max_align_t) == 0`). The authoritative fix is to strip or disallow `ORT_MEM_ADDR` when model bytes arrive over the EP trust boundary.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #435.
