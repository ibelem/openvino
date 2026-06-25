# Security finding #444: At line 294, `ext_location` is compared against the well-known sent…

**Summary:** At line 294, `ext_location` is compared against the well-known sent…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who can supply an ONNX model file can set `location="*/_ORT_MEM_ADDR_/*"`, `offset=<target_address>`, `length=<size>` to make the runtime treat an arbitrary process-virtual-address as a tensor data pointer. Downstream consumers of `m_tensor_data` (constant creation, output buffers, etc.) will read from that address, enabling information disclosure (keys, credentials, heap/stack contents) or, if the data is written back to an attacker-observable output, a full arbitrary memory read primitive. All users loading ONNX models from untrusted sources are affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file on disk → file-based model loading path (GraphIteratorProto::initialize(path) → extract_tensor_meta_info → extract_tensor_external_data)

## Description / Root cause
At line 294, `ext_location` is compared against the well-known sentinel `detail::ORT_MEM_ADDR` ("*/_ORT_MEM_ADDR_/*") with no guard checking whether the caller is a trusted in-process ORT invocation vs. a file-based load. If the match succeeds, `ext_data_offset` — parsed directly from the attacker-supplied ONNX model's `external_data[offset]` string field via `std::stoull` at line 287 — is unconditionally cast to a raw pointer at line 297: `tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset)`. No address validation, no range check, no mode check.

**Validator analysis:** The flaw is real and reachable for OpenVINO. The ORT_MEM_ADDR sentinel branch (lines 294-300) is evaluated unconditionally — before the memory_management_mode switch at line 315 and before the bounds check at lines 304-311 (which only applies to file/stream paths). When a model is loaded from disk via GraphIteratorProto::initialize(std::filesystem::path) (line 504, with the surrounding try/catch only re-throwing, not sanitizing), reset() (line 613-620) calls extract_tensor_meta_info → extract_tensor_external_data for every EXTERNAL initializer. An attacker controlling the .onnx file can set external_data location to '*/_ORT_MEM_ADDR_/*' and offset to a decimal target address; line 297 then stores that as m_tensor_data with m_is_raw=true and attacker-chosen length, and downstream constant materialization reads m_tensor_data_size bytes from the arbitrary VA — an arbitrary read / info-disclosure primitive. CWE-822 (Untrusted Pointer Dereference) is the correct classification and the impact (arbitrary memory read / info disclosure) is accurate. std::stoull at :287/:289 is itself an unguarded throw on malformed input but that is caught by the initialize try/catch, so it is only a robustness issue, not the core flaw. The proposed fix (gate the ORT_MEM_ADDR branch behind a trusted-caller flag that defaults false for file-based initialize(path)) is correct and sufficient to close the untrusted-file path; the suggested additional offset whitelist/range check has no meaningful baseline for an in-memory pointer, so it is of limited value — the trust-flag gate is the essential mitigation. A cleaner alternative: only honor ORT_MEM_ADDR when the iterator was constructed from an in-memory ModelProto (initialize(std::shared_ptr<ModelProto>), line 531), never when constructed from a path.

## Exploit / Proof of Concept
Craft a valid ONNX model with one initializer whose `data_location` flag is `EXTERNAL` and whose `external_data` map contains `{key:"location", value:"*/_ORT_MEM_ADDR_/*"}`, `{key:"offset", value:"<decimal representation of target address>"}`, `{key:"length", value:"4096"}`. When OpenVINO's ONNX front-end loads this file, `extract_tensor_external_data` is called, the sentinel matches, `m_tensor_data` is set to the attacker-chosen address, and the tensor's data is sourced from that address with no further checks.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for graph_iterator_proto.cpp:294-300 (CWE-822 untrusted pointer
// dereference). Pre-fix: an EXTERNAL initializer whose external_data location is the
// ORT_MEM_ADDR sentinel and whose offset is an attacker decimal address is accepted
// during a FILE-BASED load and reinterpret_cast to a raw uint8_t* (arbitrary VA read).
// Post-fix: file-based loading must reject the ORT_MEM_ADDR sentinel and throw.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: This requires a crafted .onnx fixture that cannot be expressed inline here,
// so this is a SKELETON. See TODOs.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // ov::frontend::onnx::tests::convert_model

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer, reject_ort_mem_addr_sentinel_on_file_load) {
    // TODO: create models/ort_mem_addr_sentinel.onnx (model.onnx.prototxt) with a single
    //       initializer:
    //         data_location: EXTERNAL
    //         external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
    //         external_data { key: "offset"   value: "4096" }   // attacker target address
    //         external_data { key: "length"   value: "4096" }
    //       and register it with the .in.cpp ${BACKEND_NAME}/model fixture mechanism.
    //
    // Pre-fix: convert_model silently builds a Constant whose data pointer == (uint8_t*)4096
    //          and reading it triggers an ASan/SEGV on an arbitrary VA.
    // Post-fix: extract_tensor_external_data throws because the ORT_MEM_ADDR branch is
    //          gated to trusted in-process callers only.
    EXPECT_THROW(convert_model("ort_mem_addr_sentinel.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='onnx_importer.reject_ort_mem_addr_sentinel_on_file_load'. Pre-fix expected failure: ASan/SEGV (heap-buffer-overflow / SEGV on unknown address ~0x1000) when the materialized Constant reads from the attacker offset, OR no throw at all (assertion failure). Post-fix: convert_model throws ov::Exception ('ORT_MEM_ADDR sentinel is not permitted in file-based model loading') and the test passes. TODO: supply the crafted models/ort_mem_addr_sentinel.onnx fixture.

## Suggested fix
Gate the `ORT_MEM_ADDR` branch on a runtime flag that is only set by trusted in-process ORT callers (e.g., a `bool allow_ort_mem_addr` parameter passed through `GraphIteratorProto`). When loading from a file path (`GraphIteratorProto::initialize(filesystem::path)`), this flag must default to `false`. Replace lines 294–300 with: `if (ext_location == detail::ORT_MEM_ADDR) { if (!graph_iterator->is_ort_mem_addr_allowed()) { throw std::runtime_error("ORT_MEM_ADDR sentinel is not permitted in file-based model loading"); } ... }`. Additionally, add a whitelist/range validation of `ext_data_offset` even for trusted callers.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #444.
