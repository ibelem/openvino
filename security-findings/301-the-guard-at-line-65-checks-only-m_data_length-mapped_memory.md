# Security finding #301: The guard at line 65 checks only `m_data_length > mapped_memory->si…

**Summary:** The guard at line 65 checks only `m_data_length > mapped_memory->si…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition / CWE-125: Out-of-bounds Read
**Severity / Impact:** Any subsequent read from the resulting SharedBuffer dereferences memory beyond the mmap region, triggering a SIGSEGV (crash/DoS) or, if adjacent memory is mapped, an information leak or memory corruption. Affects any application loading untrusted ONNX models with external data via the mmap path.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model external-data file: the `offset` and `length` fields in TensorProto external_data, plus the referenced file on disk

## Description / Root cause
The guard at line 65 checks only `m_data_length > mapped_memory->size()` but never checks `m_offset < mapped_memory->size()`. The stat-based size check at lines 53-55 uses `file_size` from a `stat()` call (line 52), but the actual mmap at line 62 is a separate OS call. If the file is truncated (or replaced) between the stat and the mmap, `mapped_memory->size()` will be smaller than `file_size`. An attacker-controlled `m_offset` that was accepted by the stat-based guard (line 53) can then satisfy `m_offset >= mapped_memory->size()`, making `mapped_memory->data() + m_offset` at line 69 an out-of-bounds pointer past the end of the mapped region.

**Validator analysis:** The cited code has two independent bounds gates: lines 52-55 validate both m_offset and m_data_length against the stat-derived file_size, while lines 65-67 (after the actual mmap) re-validate only m_data_length against mapped_memory->size() and never re-validate m_offset against the mapped size. Under normal conditions mapped_memory->size()==file_size, so the line 52-55 gate already protects line 69, and no OOB occurs (worst case is a zero-length one-past-end pointer when m_offset==file_size && m_data_length==0). The flaw is therefore real ONLY through the TOCTOU race the finding describes: if the attacker-controlled external-data file is truncated/swapped between file_size() (line 52) and load_mmap_object() (line 62), mapped_memory->size() shrinks below file_size, an m_offset that passed the stale check can exceed the live mapping, and line 69 forms an OOB pointer while line 70's fallback length uses the stale file_size — an out-of-bounds SharedBuffer. The CWE-367 + CWE-125 classification is accurate; DoS/info-leak impact is plausible. The proposed fix is correct and sufficient: deriving ALL bounds from mapped_memory->size() (reject when size()==0 || m_offset>=size() || m_data_length>size()-m_offset, and use size()-m_offset for the fallback length) makes the post-mmap check self-consistent and closes the race regardless of stale stat data. One nit: to also accept the legitimate m_offset==size() && m_data_length==0 (zero-byte tail) case, the check could use m_offset>size() with a separate zero-length guard, but rejecting it is harmless. A deterministic unit test cannot reproduce the race without filesystem interposition, so only a skeleton is provided.

## Exploit / Proof of Concept
Attacker provides an ONNX model whose external_data `offset` field equals N (e.g., 0x1000) and `length` == 0. The referenced data file initially has size > N (passes stat check at line 53). Between the stat call (line 52) and `ov::load_mmap_object` (line 62), the file is truncated to < N bytes (race via symlink swap or cooperative filesystem). The guard at line 65 only checks `m_data_length(0) > mapped_memory->size()` — false — so it passes. Line 69 computes `mapped_memory->data() + 0x1000` past the end of the (now smaller) mapping. When the tensor data is accessed, the process reads unmapped memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for TOCTOU/OOB at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65-71
// Pre-fix: the post-mmap guard (line 65) checks only m_data_length vs mapped size and
//   omits m_offset < mapped_size, and line 70 uses the stale stat() file_size for the
//   fallback length. If the external-data file shrinks between file_size() (line 52)
//   and load_mmap_object() (line 62), line 69 (mapped->data()+m_offset) is OOB.
// Post-fix: all bounds derived from mapped_memory->size(); load is rejected with
//   ov::Exception / onnx::error::invalid_external_data.
//
// Lives in openvino/src/frontends/onnx/tests in the style of onnx_import.in.cpp,
// target ov_onnx_frontend_tests (built with ASan).

#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // FrontEndTestUtils convert_model(...)
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: This race cannot be triggered deterministically from a static .onnx fixture
//       because under non-race conditions mapped_memory->size() == file_size and the
//       line 52-55 stat-based guard already rejects an out-of-range offset. Reproducing
//       requires interposing the filesystem (truncate/symlink-swap the external data
//       file between stat() and mmap()).
//
// TODO: Provide a crafted model 'external_data_offset_past_mmap.onnx' whose TensorProto
//       external_data sets offset=N (e.g. 0x1000), length=0, location pointing at a
//       helper-controlled file, AND a test harness that shrinks that file below N after
//       file_size() but before load_mmap_object() (e.g. via an LD_PRELOAD/syscall shim
//       or a FUSE/cooperative filesystem). Without that interposition this assertion
//       cannot fail pre-fix.
//
// TODO: Confirm the exact convert_model helper signature + fixture path from
//       onnx_import.in.cpp before enabling.
TEST(onnx_external_data, DISABLED_offset_past_truncated_mmap_is_rejected) {
    // Expected post-fix behaviour: even if the mapped region is smaller than the
    // stat()-reported size, an offset that lands outside the live mapping must throw.
    EXPECT_THROW(convert_model("external_data/external_data_offset_past_mmap.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests (with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_onnx_frontend_tests --gtest_filter='onnx_external_data.*offset_past_truncated_mmap*'. The test is DISABLED_ and a skeleton: it needs (1) a crafted external_data .onnx fixture with offset past the file and length=0, and (2) a filesystem interposition shim that truncates the data file between file_size() and load_mmap_object(). Pre-fix the access at tensor_external_data.cpp:69 yields an ASan heap/SEGV 'AddressSanitizer: SEGV on unknown address' or out-of-bounds read on the mmap region; post-fix convert_model throws ov::Exception (invalid_external_data) and the test passes.

## Suggested fix
Replace the guard at line 65 with a combined check that uses `mapped_memory->size()` exclusively (not `file_size`) for all bounds: `if (mapped_memory->size() == 0 || m_offset >= mapped_memory->size() || m_data_length > mapped_memory->size() - m_offset) { throw error::invalid_external_data{*this}; }`. Then on line 70, use `mapped_memory->size() - m_offset` instead of `static_cast<uint64_t>(file_size) - m_offset` so the buffer length is bounded by the actual mapping.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #301.
