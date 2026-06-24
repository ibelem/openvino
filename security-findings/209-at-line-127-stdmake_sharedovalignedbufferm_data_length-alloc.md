# Security finding #209: At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` a…

**Summary:** At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` a…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** An attacker can set `length` to a value like `0xFFFFFFFFFFFFFFFF` causing `AlignedBuffer` to attempt allocation of ~18 EB of memory, crashing the host process with `std::bad_alloc` or exhausting virtual address space (DoS). In a server context this affects all concurrent inference sessions.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file (protobuf) → TensorProto::external_data `length` entry → m_data_length (constructor line 26)

## Description / Root cause
At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` allocates a buffer whose size is the raw `uint64_t` value parsed from the protobuf `length` field (constructor line 26 via `std::stoull`). There is no upper-bound validation on `m_data_length` before this allocation. The same absence of a bound also applies to the `is_empty_buffer` path: if `m_offset != 0` and `m_data_length == 0` the allocation is zero bytes (benign), but any non-zero length is accepted.

**Validator analysis:** The CWE-789 claim is accurate for load_external_mem_data: unlike the file-backed paths (lines 53-55 and 83-85) which cross-check m_data_length against the actual file size, the ORT_MEM_ADDR path has NO size bound at all — is_valid_buffer only requires m_offset && m_data_length to be non-zero (line 121), so any uint64_t length is accepted and passed to AlignedBuffer (line 127), and AlignedBuffer::AlignedBuffer (aligned_buffer.cpp:18-19) performs an unchecked util::aligned_alloc. A length like 0xFFFFFFFFFFFFFFFF throws std::bad_alloc; a more moderate huge value (e.g. tens of GB) attempts a real allocation and can OOM-kill the process => DoS, so the impact stands. NOTE: this path is actually MORE dangerous than the finding states — line 129 memcpy's m_data_length bytes FROM reinterpret_cast<char*>(m_offset), an attacker-controlled pointer (arbitrary read / crash), which is a separate, more severe defect outside this finding's scope. The proposed fix (validate m_data_length against an upper bound before allocation) is correct but insufficient on its own: the same guard plus validation of m_offset as a legitimate registered shared-memory region is required to address the arbitrary-pointer memcpy. For openvinoEp I reject only on unproven reachability, not because the defect is unreal. A clean exception (bad_alloc) caught by the frontend would partly mitigate the 18EB case, but intermediate large values are a genuine DoS, so validated for openvino.

## Exploit / Proof of Concept
Same crafted ONNX as the primary finding, but with `length` set to a large decimal string such as `"18446744073709551615"`. The `std::stoull` at constructor line 26 succeeds (no overflow — it stores into `uint64_t`). `load_external_mem_data` line 127 passes this to `AlignedBuffer`, which calls `::operator new`, triggering OOM.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
// (TensorExternalData::load_external_mem_data). Pre-fix: an external_data tensor whose
// `location` == ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") with a huge `length` reaches
// std::make_shared<ov::AlignedBuffer>(m_data_length) with no upper-bound check, throwing
// std::bad_alloc / OOM-killing the process. Post-fix: an out-of-range length must be
// rejected with ov::Exception (error::invalid_external_data) instead of bad_alloc/OOM.
//
// SKELETON: a crafted .onnx fixture is required (proto-level external_data entries cannot be
// expressed inline in this harness), so the symbols/fixture below are placeholders.
#include "onnx_utils.hpp"          // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/test_case.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace from onnx_import.in.cpp

// TODO: create models/external_data/ort_mem_addr_huge_length.onnx :
//   a TensorProto with data_location = EXTERNAL and external_data entries:
//     location = "*/_ORT_MEM_ADDR_/*"   (detail::ORT_MEM_ADDR)
//     offset   = "4096"                 (non-zero so is_valid_buffer is true)
//     length   = "18446744073709551615" (UINT64_MAX -> excessive allocation)
TEST(onnx_external_data, ort_mem_addr_excessive_length_is_rejected) {
    // Pre-fix this constructs AlignedBuffer(UINT64_MAX) -> std::bad_alloc (NOT ov::Exception),
    // i.e. the test fails. Post-fix the length is bounds-checked and converted to ov::Exception.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_huge_length.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.ort_mem_addr_excessive_length_is_rejected . Provide the crafted fixture models/external_data/ort_mem_addr_huge_length.onnx (TensorProto external_data location="*/_ORT_MEM_ADDR_/*", offset="4096", length="18446744073709551615"). Expected pre-fix: std::bad_alloc / ASan allocation-size-too-big abort escapes EXPECT_THROW(ov::Exception) (test fails). Post-fix: bounded length throws error::invalid_external_data (ov::Exception) and the test passes.

## Suggested fix
Before the allocation at line 127, validate `m_data_length` against a reasonable upper bound (e.g., a configurable max tensor size, or the known size of the registered shared-memory region). At minimum: `if (m_data_length > MAX_ALLOWED_TENSOR_BYTES) { throw error::invalid_external_data{*this}; }` where `MAX_ALLOWED_TENSOR_BYTES` is a compile-time or runtime limit. The same guard should also be applied in `load_external_data()` and `load_external_mmap_data()` for the file-backed paths, although those are partially protected by the file-size cross-check at lines 83-85 and 53-55 respectively.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #209.
