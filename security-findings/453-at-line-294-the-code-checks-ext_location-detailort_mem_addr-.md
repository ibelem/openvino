# Security finding #453: At line 294 the code checks `ext_location == detail::ORT_MEM_ADDR` …

**Summary:** At line 294 the code checks `ext_location == detail::ORT_MEM_ADDR` …

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who can supply a crafted ONNX model sets `m_tensor_data` to an arbitrary attacker-chosen pointer. Downstream code that dereferences the pointer for tensor data reads (or writes) will access arbitrary process memory, enabling information disclosure, crash/DoS, or potential RCE depending on how the pointer is later used. Affects any caller of OpenVINO's ONNX frontend that processes externally-provided models.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:297` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied ONNX model file (TensorProto.external_data entries) crossing into extract_tensor_external_data

## Description / Root cause
At line 294 the code checks `ext_location == detail::ORT_MEM_ADDR` (the string "*/_ORT_MEM_ADDR_/*"), and if true, at line 297 it executes `tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset)`. The value `ext_data_offset` is populated at line 287 via `std::stoull(entry.value())` — a completely attacker-controlled string from the protobuf payload — with no range or alignment check. Any crafted ONNX model can reach this branch and plant an arbitrary address into `m_tensor_data`.

**Validator analysis:** The defect is real in openvino. extract_tensor_external_data parses the fully attacker-controlled external_data repeated field of a TensorProto (lines 283-292). The ORT_MEM_ADDR branch (lines 294-300) is selected solely by string-equality on the attacker-controlled location value ("*/_ORT_MEM_ADDR_/*", tensor_external_data.hpp:91); there is no caller-context gate distinguishing an in-process ORT hand-off from a model read off disk. ext_data_offset (std::stoull of the attacker's 'offset' string, line 287) is then reinterpret_cast to uint8_t* and stored in m_tensor_data with attacker-chosen size (line 298). This branch is reached unconditionally from extract_tensor_meta_info (line 396) for any tensor with data_location==EXTERNAL, including via OpenVINO's public ONNX frontend reading an arbitrary model file (core.read_model). Downstream consumption of m_tensor_data/m_tensor_data_size to materialize constant data dereferences the planted pointer => arbitrary read/crash, matching CWE-822 (Untrusted Pointer Dereference); impact (info disclosure / DoS, possibly worse) is accurate. Note the file-backed paths below (lines 302-311) DO perform file_size bounds checks, underscoring that the ORT_MEM_ADDR branch is the unguarded outlier (the Internal_MMAP/Stream branch at line 364 does the same raw cast but only after the 304-305 bounds check). The proposed fix — a caller-supplied allow_ort_mem_addr flag defaulting false, gating lines 294-300 — is correct and sufficient: only an explicit in-process ORT caller should enable the shared-memory address path; a model loaded from a file path must never honor the marker. The fallback suggestion (validate the address lies in a registered region and length is plausible) is a reasonable defense-in-depth but the context flag is the primary fix. openvinoEp is the legitimate (trusted) producer of this marker and does not host the vulnerable code, so it is rejected for this finding.

## Exploit / Proof of Concept
Craft an ONNX model with a TensorProto whose `data_location` is `EXTERNAL` and whose `external_data` repeated field contains entries `{key:"location", value:"*/_ORT_MEM_ADDR_/*"}` and `{key:"offset", value:"<target address as decimal>"}`. When the model is loaded, `extract_tensor_external_data` hits the ORT_MEM_ADDR branch; `ext_data_offset` becomes the attacker-chosen integer; `reinterpret_cast<uint8_t*>(ext_data_offset)` stores it as a raw pointer. No prior check prevents reaching line 297 other than the string equality on the location field, which the attacker controls.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-300
// (extract_tensor_external_data). An attacker-supplied ONNX model whose tensor
// is data_location=EXTERNAL with external_data {location:"*/_ORT_MEM_ADDR_/*",
// offset:"<decimal addr>", length:"<n>"} currently reaches
//   m_tensor_data = reinterpret_cast<uint8_t*>(stoull(offset));
// with no context gate or validation, planting an arbitrary pointer.
//
// This assertion encodes the fix: a model loaded from a *file* (not an in-process
// ORT hand-off) must NOT honor the ORT_MEM_ADDR marker. Pre-fix the offset is
// cast to a raw pointer and later dereferenced (ASan: SEGV / heap-buffer-overflow
// on read of arbitrary address); post-fix convert_model rejects the model.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected_from_file) {
    // TODO: provide crafted fixture models/external_data/ort_mem_addr_arbitrary.onnx
    //       with a single initializer:
    //         data_location: EXTERNAL
    //         external_data: { key:"location" value:"*/_ORT_MEM_ADDR_/*" }
    //         external_data: { key:"offset"   value:"4096" }   // arbitrary addr
    //         external_data: { key:"length"   value:"64" }
    //       (a protobuf-serialized .onnx must be added under the test data dir;
    //        a self-contained in-memory build is not possible because
    //        extract_tensor_external_data is in an anonymous namespace.)
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_ort_mem_addr_rejected_from_file*'. Add the crafted fixture .onnx (TODO above) to the frontend test data dir. Expected pre-fix under ASan: SEGV / unknown-address read when the planted reinterpret_cast<uint8_t*>(offset) pointer is dereferenced during constant materialization; expected post-fix: convert_model throws ov::Exception and the test passes.

## Suggested fix
The ORT shared-memory path should only be reachable when the model was explicitly provided by ORT in-process (not from a file). Add a caller-supplied flag (e.g. `allow_ort_mem_addr`) that defaults to `false` and gate the entire block lines 294–300 on it. If ORT interop must remain unconditional, at minimum validate that `ext_data_offset` falls within a previously registered/allowed memory region, and that `ext_data_length` is non-zero and plausible, before performing the cast.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #453.
