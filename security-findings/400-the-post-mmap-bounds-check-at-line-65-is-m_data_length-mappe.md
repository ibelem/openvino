# Security finding #400: The post-mmap bounds check at line 65 is `m_data_length > mapped_me…

**Summary:** The post-mmap bounds check at line 65 is `m_data_length > mapped_me…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition / CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read (and potential crash/SIGSEGV) when the returned SharedBuffer is subsequently accessed by the ONNX frontend. On Linux the first access to the out-of-range pointer after mmap boundary causes SIGBUS/SIGSEGV, crashing the inference process. On a shared NFS/FUSE mount, an attacker co-located with the model files can trigger this deterministically to achieve denial of service or, on platforms where adjacent pages are readable (e.g., large over-mmap), an information leak from process memory.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX TensorProto external_data 'offset' and 'length' protobuf fields parsed at line 24 via std::stoull; combined with an attacker-accessible external data file path.

## Description / Root cause
The post-mmap bounds check at line 65 is `m_data_length > mapped_memory->size() || mapped_memory->size() == 0`. It does NOT validate that `m_offset + m_data_length <= mapped_memory->size()` (i.e., that `m_offset` itself is within the mapping). The pre-mmap guard at lines 52-56 uses `ov::util::file_size(full_path)` (a stat call), but the actual mapping is created at line 62 (`ov::load_mmap_object`). Between these two calls an attacker-controlled file can be truncated/swapped so that `mapped_memory->size() < m_offset` while still `mapped_memory->size() >= m_data_length`, causing line 65's check to pass. The pointer arithmetic at line 69 (`mapped_memory->data() + m_offset`) then produces a pointer past the end of the mapped region.

**Validator analysis:** The defect is real: line 65's post-mmap guard is strictly weaker than the pre-mmap guard at 53-54. The pre-check validates `m_offset > file_size - m_data_length` (i.e. offset+length <= file_size) against a stat'd size, but the post-check only validates `m_data_length > mapped_memory->size()` and never re-validates `m_offset` against the actual mapped size. With a large `m_offset` and small `m_data_length`, line 65 passes while `m_offset` exceeds `mapped_memory->size()`, so line 69's pointer arithmetic yields an out-of-bounds base pointer wrapped in the returned SharedBuffer. The TOCTOU window (stat at l.52 vs mmap at l.62, plus a cache path at 57-63 that reuses a previously-mapped object while re-stat'ing the file) makes the two sizes divergeable by a co-located attacker who truncates/replaces the external-data file; first access to the past-end pointer triggers SIGBUS/SIGSEGV on Linux. CWE-367 (TOCTOU) is the mechanism and CWE-125 (OOB read) the consequence — both accurate; DoS impact is sound, info-leak is more speculative. The proposed fix is correct and sufficient: it mirrors the pre-check against `mapped_memory->size()` (`size==0 || m_offset > size || m_data_length > size - m_offset`), bounding both offset and offset+length against the value taken from the mapping itself and closing the window. One refinement: also guard the `m_data_length == 0` branch at line 70 (which falls back to `file_size - m_offset` from the possibly-stale stat) so the computed length likewise uses `mapped_memory->size() - m_offset`. Note this is a defense-in-depth/race defect: in the no-race common case mapped size == file_size and the path is safe, but the missing offset re-validation is a genuine omission.

## Exploit / Proof of Concept
1. Craft an ONNX model with external_data offset=0x100000 and length=16. 2. Provide a legitimate external data file of size >= 0x100010 so the pre-check at lines 53-54 passes. 3. Race: after the file_size() call at line 52 returns and before load_mmap_object() at line 62 executes, truncate (or atomically replace via symlink swap) the file to 64 bytes. 4. load_mmap_object returns a mapping of 64 bytes; mapped_memory->size()=64 >= m_data_length=16 so line 65 does NOT throw. 5. Line 69: `mapped_memory->data() + 0x100000` is past the 64-byte mapping — OOB pointer is wrapped in the returned SharedBuffer. 6. When the ONNX frontend dereferences the buffer to extract tensor data, SIGSEGV or silent OOB read occurs.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for tensor_external_data.cpp:65 (CWE-367/CWE-125).
// Pre-fix: load_external_mmap_data() only checks `m_data_length > mapped_memory->size()`,
//   so a mapping whose actual size is smaller than m_offset (file shrank between the
//   file_size() stat at l.52 and load_mmap_object() at l.62) still passes line 65 and
//   line 69 computes `mapped_memory->data() + m_offset` past the mapping end -> OOB/SIGBUS.
// Post-fix: the combined guard rejects m_offset > mapped_size, so load throws ov::Exception.
//
// NOTE: this defect is race/state dependent — a fully self-contained deterministic test
// requires simulating the truncation between stat and mmap (or seeding the mmap cache with
// a smaller mapping than the on-disk file). The harness hooks below are TODOs.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
#include "gtest/gtest.h"
// TODO: include the real frontend test fixture headers used by onnx_import.in.cpp
//       (e.g. "common_test_utils/test_control.hpp", "onnx_utils.hpp").

using namespace ov::frontend;

TEST(onnx_external_data, mmap_offset_past_mapped_size_is_rejected) {
    // TODO: build an .onnx whose TensorProto external_data has
    //       offset=0x100000 (>> file), length=16, location="data.bin".
    // TODO: place a 'data.bin' that is >= 0x100010 bytes for the pre-stat check (l.53-54)
    //       to pass, then arrange for the mmap to observe a truncated (e.g. 64-byte) file
    //       — e.g. by pre-populating the MappedMemoryHandles cache (l.57-63) with a 64-byte
    //       mapping of the same path so file_size() and mapped_memory->size() diverge.
    // TODO: invoke the mmap-enabled load path (FrontEnd::load with mmap=true ->
    //       TensorExternalData::load_external_mmap_data).
    //
    // Expected after fix: throws because m_offset (0x100000) > mapped_memory->size() (64).
    EXPECT_THROW(
        { /* TODO: convert_model("crafted_external_offset.onnx") with mmap weights */ },
        ov::Exception);
    // Pre-fix this returns a SharedBuffer over an out-of-bounds pointer; first deref -> ASan/SIGBUS.
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests. Run: ./ov_onnx_frontend_tests --gtest_filter=onnx_external_data.mmap_offset_past_mapped_size_is_rejected (run under ASan). Pre-fix expectation: AddressSanitizer/SEGV (heap/mmap out-of-bounds) or no throw when the SharedBuffer over `mapped_memory->data()+m_offset` (l.69) is dereferenced; post-fix expectation: ov::Exception thrown by the combined offset/length guard at line 65.

## Suggested fix
Replace line 65's guard with a complete combined check that mirrors the pre-check logic against the actual mapped size:

  if (mapped_memory->size() == 0 ||
      m_offset > mapped_memory->size() ||
      m_data_length > mapped_memory->size() - m_offset) {
      throw error::invalid_external_data{*this};
  }

This ensures both m_offset and m_offset+m_data_length are within [0, mapped_memory->size()] using values from the mapping itself, eliminating the TOCTOU window entirely.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #400.
