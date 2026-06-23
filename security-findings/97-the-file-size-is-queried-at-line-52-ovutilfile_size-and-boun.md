# Security finding #97: The file size is queried at line 52 (`ov::util::file_size`), and bo…

**Summary:** The file size is queried at line 52 (`ov::util::file_size`), and bo…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** File-size validation is rendered ineffective. An attacker who can substitute the external data file between `file_size()` and `load_mmap_object()` can cause the mmap to be smaller than the validated bounds, leading to OOB reads (CWE-125) when tensor data is accessed — crash/DoS or potential information disclosure.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:52` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied ONNX model file specifies an external data path; the file at that path is on a filesystem the attacker influences (e.g., NFS mount, symlink, or shared model directory)

## Description / Root cause
The file size is queried at line 52 (`ov::util::file_size`), and bounds are validated against that snapshot. However, the actual memory mapping is obtained separately — either from a potentially stale cache (line 60) or via a new `ov::load_mmap_object` call (line 62). Between the `stat` at line 52 and the `mmap` at line 62, an attacker on the same host (or controlling the filesystem) can replace `data.bin` with a smaller file. The mapping then reflects the smaller file, while bounds were computed against the larger one. Additionally, even without a race in the new-mapping branch, the cache path (line 60) always uses a mapping from a prior moment in time, creating an unconditional TOCTOU for repeated loads.

**Validator analysis:** Confirmed real for openvino. Lines 52-56 validate m_offset/m_data_length against file_size (a stat snapshot). The post-mapping guard at lines 65-67 is the right idea but incomplete: it checks `m_data_length > mapped_memory->size()` and `size()==0`, yet never re-checks the offset (`m_offset` / `m_offset + m_data_length`) against the *actual* mapping size. Therefore if mapped_memory->size() < file_size — via the TOCTOU window between file_size() (l.52) and load_mmap_object() (l.62), or via a stale cache entry (l.59-60) whose underlying file grew since it was first mmapped — a large m_offset that passed validation against the bigger file_size will index past the smaller mapping at line 69 (`mapped_memory->data() + m_offset`), and the zero-length branch at line 70 (`file_size - m_offset`) compounds it by using file_size, not the mapping size. The primary consequence is an OOB read (CWE-125 / DoS / info-disclosure); the TOCTOU label (CWE-367) is one reachability vector but the deeper, label-independent defect is that bounds are not re-anchored to the authoritative mapping size. The proposed fix is correct and sufficient: after obtaining the mapping, validate `mapped_memory->size()==0 || m_data_length > mapped_memory->size() || m_offset > mapped_memory->size() - m_data_length`, and additionally the line-70 zero-length fallback should use `mapped_memory->size() - m_offset` rather than `file_size - m_offset` so the SharedBuffer length is also anchored to the real mapping. Reachability from the ONNX-model boundary alone is limited (file_size and mmap normally read the same bytes), but within the stated boundary (attacker-influenced filesystem / shared model dir / file growth between cached loads) it is reachable; that justifies validated rather than rejected.

## Exploit / Proof of Concept
Attacker symlinks or replaces `data.bin` with a 1-byte file immediately after the `file_size` stat at line 52 returns a large value. The subsequent `load_mmap_object` at line 62 maps only 1 byte, but the returned SharedBuffer uses `m_offset` and `m_data_length` that were validated against the larger size, pointing well past the 1-byte mapping.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 OOB read in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65-71
// The post-mmap guard at line 65 omits the m_offset bound, so a mapping smaller
// than the file_size validated at lines 52-54 lets SharedBuffer index past the
// mapping at line 69 (data()+m_offset) / line 70 (file_size - m_offset).
//
// SKELETON: a deterministic, self-contained gtest cannot be authored cleanly
// because triggering the flaw requires mapped_memory->size() to diverge from
// the file_size stat — i.e. either the TOCTOU file-swap race between l.52 and
// l.62, or a cache entry (l.59-60) whose backing file grew after being mmapped.
// Neither is reproducible from a single static .onnx + data file in the normal
// ov_onnx_frontend_tests flow (there file_size == mapping size, so the bug is
// masked). The items below name exactly what is missing.
//
// Suggested location: src/frontends/onnx/tests/onnx_import.in.cpp
// (use the existing convert_model() helper as in that file).

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // for convert_model(...) used across onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

TEST(${BACKEND_NAME}, onnx_external_data_offset_oob_after_mapping_shrinks) {
    // TODO(fixture): provide a crafted model 'external_data_offset_oob.onnx'
    //   whose tensor external_data sets a LARGE offset+length that is valid
    //   against the INITIAL data file size, e.g. offset=900, length=50,
    //   data file 'data.bin' initially 1000 bytes.
    // TODO(race/grow): after convert begins (or via a pre-cached mapping whose
    //   file later shrank to 100 bytes), the mmap must be smaller than the
    //   validated file_size. There is no clean public hook to force this in
    //   the frontend test harness; a unit test on TensorExternalData directly
    //   (constructing it with offset=900,length=50 and pointing at a 100-byte
    //   file while a stale 1000-byte mapping sits in the MappedMemoryHandles
    //   cache) is the realistic encoding. That requires friending/exposing
    //   load_external_mmap_data + a hand-built cache map.
    //
    // Expected with the fix in place: the offset re-check at the post-mapping
    // guard rejects the load -> convert_model throws.
    // Pre-fix (ASan build): heap-buffer-overflow READ in
    //   ov::SharedBuffer ctor / subsequent tensor access (data()+m_offset).
    EXPECT_THROW(convert_model("external_data_offset_oob.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_offset_oob_after_mapping_shrinks*. Pre-fix expectation under ASan: heap-buffer-overflow on READ originating from ov::SharedBuffer over (mapped_memory->data() + m_offset) at tensor_external_data.cpp:69-70, because m_offset was validated against the larger file_size, not the smaller mapped_memory->size(). Post-fix: the added offset bound at line ~65 throws ov::Exception (error::invalid_external_data) and the test passes. NOTE: requires the TODO crafted fixture + a way to force mapping size < file_size (stale cache / file shrink); not achievable from a single static fixture, hence skeleton.

## Suggested fix
After obtaining the mapping (whether from cache or fresh), revalidate all bounds against `mapped_memory->size()` rather than the separately-queried `file_size`. The corrected guard should be placed after line 64 and should check: `mapped_memory->size() == 0 || m_data_length > mapped_memory->size() || m_offset > mapped_memory->size() - m_data_length`. This makes `mapped_memory->size()` the single authoritative source of truth, eliminating both the TOCTOU race and the stale-cache OOB.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #97.
