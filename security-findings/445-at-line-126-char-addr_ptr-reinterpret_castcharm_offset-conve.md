# Security finding #445: At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` c…

**Summary:** At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` c…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Same arbitrary memory read as Finding 1, but triggered through the `TensorExternalData::load_external_mem_data` code path. An oversized `m_data_length` additionally risks `std::bad_alloc` (DoS) at the `AlignedBuffer` allocation (line 127) or, if allocation succeeds, copies a large chunk of process memory into a buffer that may be returned to or inspected by the caller.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file on disk → TensorExternalData constructor (line 24, m_offset = std::stoull(entry.value())) → load_external_mem_data

## Description / Root cause
At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` converts the model-supplied `offset` integer directly to a raw pointer. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads `m_data_length` (also attacker-controlled) bytes from that address. There is no validation that `m_offset` falls within any legitimate memory region, and `m_data_length` has no upper bound check before the allocation at line 127 or the memcpy at line 129.

**Validator analysis:** The flaw is real and reachable in openvino. Unlike load_external_data (lines 83-84) and load_external_mmap_data (lines 53-54), which bounds-check offset+length against the on-disk file size, load_external_mem_data has NO bounds/region validation — it trusts that the ORT_MEM_ADDR sentinel implies a legitimate in-process address. But the dispatch decision in tensor.hpp:324 is driven solely by the location string read from the (untrusted) ONNX TensorProto, so a crafted file can spoof the sentinel and supply an arbitrary m_offset and m_data_length, yielding an attacker-controlled memcpy source address and size (CWE-125 / arbitrary read; oversized length also risks bad_alloc DoS at line 127). vulnType and impact are accurate. The proposed fix (cap m_data_length + validate addr against registered regions + gate the sentinel to trusted callers) is directionally correct but the ROOT fix is that the in-memory-vs-file decision must NOT be derived from an attacker-controlled location string embedded in the model — it should come from a trusted flag/API parameter set only by the in-process ORT caller. Without that, length caps and region checks are incomplete. For openvinoEp the marker is produced internally with valid pointers, so the boundary cannot supply a raw attacker address through it — rejected.

## Exploit / Proof of Concept
Same malicious ONNX file as Finding 1. When `load_external_mem_data` is called (e.g., from the `TensorExternalData`-based loading path), `m_offset` holds the attacker-chosen address and `m_data_length` holds the read size. The `is_valid_buffer` check at line 121 only ensures both are non-zero, not that they are valid. The memcpy at line 129 then reads `m_data_length` bytes starting at the attacker-supplied address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:126-129 (load_external_mem_data):
//   a file-based ONNX model must NOT be able to spoof the ORT_MEM_ADDR sentinel
//   ("*/_ORT_MEM_ADDR_/*") to coerce reinterpret_cast<char*>(m_offset) + memcpy.
// Pre-fix: convert_model dereferences the attacker offset (ASan: SEGV / heap-buffer-overflow
//   on the memcpy source at tensor_external_data.cpp:129) or crashes.
// Post-fix: the file-based path rejects the spoofed sentinel and throws ov::Exception.
//
// Style mirrors onnx_import.in.cpp in ov_onnx_frontend_tests.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: provide a crafted fixture models/external_data/ort_mem_addr_spoof.onnx where an
//       initializer has data_location=EXTERNAL and external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"   (the ORT_MEM_ADDR sentinel)
//         offset   = <a bogus/non-mappable address, e.g. 0xdeadbeef>
//         length   = <large value, e.g. 0x40000000>
//       A pure-from-disk ONNX model must never reach load_external_mem_data().
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_rejects_spoofed_ort_mem_addr) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_spoof.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_rejects_spoofed_ort_mem_addr* (built with -DENABLE_SANITIZER=ON for ASan). Pre-fix the crafted model reaches load_external_mem_data and ASan reports SEGV / heap-buffer-overflow at tensor_external_data.cpp:129 (memcpy from reinterpret_cast<char*>(m_offset)); post-fix convert_model throws ov::Exception (invalid_external_data) and the test passes. NOTE: requires the crafted .onnx fixture noted in the TODO.

## Suggested fix
Before calling `load_external_mem_data`, verify that the data location sentinel is only permitted for in-process/trusted callers (see Finding 1 fix). Within the function itself, add an upper-bound cap on `m_data_length` (e.g., refuse values above a sane maximum like 2 GB), and validate that `addr_ptr` is non-null and falls within a pre-registered set of valid shared-memory regions supplied by the trusted ORT caller. Never allow this function to be reached from a file-based loading context.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #445.
