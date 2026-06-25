# Security finding #401: When `m_data_length == 0` (the field is absent in the protobuf), li…

**Summary:** When `m_data_length == 0` (the field is absent in the protobuf), li…

**CWE IDs:** CWE-367: TOCTOU / CWE-805: Buffer Access with Incorrect Length Value
**Severity / Impact:** Same crash/OOB-read impact as finding 1, but triggered without needing a large offset — any model that omits the 'length' field (valid per the ONNX spec: length=0 means 'read to end of file') combined with a file shrink between stat and mmap is sufficient.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:70` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX external_data 'length'=0 (default/absent) combined with an attacker-shrinkable external data file.

## Description / Root cause
When `m_data_length == 0` (the field is absent in the protobuf), line 70 computes the read length as `static_cast<uint64_t>(file_size) - m_offset`, where `file_size` is the stale value captured at line 52 before the mmap. After a TOCTOU shrink, `mapped_memory->size()` can be smaller than `file_size`, making this computed length exceed `mapped_memory->size() - m_offset`. The SharedBuffer is then constructed with a length that extends past the mapping.

**Validator analysis:** The defect is real for the m_data_length==0 branch. At line 52 file_size is captured by stat, the file is mmap'd at line 62, and line 70 derives the buffer length from the stale file_size when the protobuf 'length' field is absent (m_data_length==0). The post-mmap guard at line 65 (`m_data_length > mapped_memory->size() || mapped_memory->size()==0`) is a no-op for the length value when m_data_length==0 — it never compares m_offset, nor the computed `file_size - m_offset`, against mapped_memory->size(). Thus if the file is truncated between stat and mmap (TOCTOU), mapped_memory->size() < file_size and the SharedBuffer claims `file_size - m_offset` bytes over a smaller mapping → OOB read. The CWE-367/CWE-805 classification is accurate; impact is OOB read/crash, not write, so 'OOB-read' is the correct severity framing (no memory corruption/RCE established). Exploitability is genuine but narrow: it requires an attacker able to concurrently shrink the external-data file during model load (the stated trust boundary). The proposed fix (compute effective_length from mapped_memory->size() - m_offset) is correct and necessary but INSUFFICIENT alone: line 65 must also reject m_offset > mapped_memory->size() (and m_offset > 0 with size==0) to prevent unsigned underflow of `mapped_memory->size() - m_offset`. Recommend: after mmap, validate `m_offset <= mapped_memory->size()` and `(m_data_length==0 ? 0 : m_data_length) <= mapped_memory->size() - m_offset`, then set effective_length = m_data_length>0 ? m_data_length : mapped_memory->size() - m_offset. The reachability from the ORT OpenVINO EP is not established (na for openvinoEp); only OpenVINO's own ONNX frontend reaches this code.

## Exploit / Proof of Concept
Provide an ONNX model with external_data containing only 'location' and 'offset'=0 (no 'length' field, so m_data_length stays 0). Supply a 1 MB external data file so the pre-check passes. Race: truncate the file to 512 bytes after line 52 but before line 62. mapped_memory->size()=512; line 65 check: `0 > 512` is false and `512 == 0` is false — check passes. Line 70: `file_size(=1MB) - 0 = 1MB`; SharedBuffer is created claiming 1 MB of data at a 512-byte mapping → every access past byte 512 is OOB.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression for tensor_external_data.cpp:70 (load_external_mmap_data):
//   when the ONNX external_data 'length' field is absent (m_data_length==0),
//   the read length must be derived from the ACTUAL mapped size, not the stale
//   pre-mmap stat (file_size). Pre-fix, a model whose external-data file is
//   smaller than the stat-time size (TOCTOU shrink) yields a SharedBuffer that
//   overruns the mapping -> OOB read (ASan heap/anon-mapping over-read).
//
// NOTE: This defect is fundamentally a TOCTOU race (file shrunk between
//   ov::util::file_size() at line 52 and ov::load_mmap_object() at line 62),
//   which cannot be triggered deterministically from a static gtest without
//   injecting a file-shrink between those two calls. Therefore this is a
//   SKELETON: it shows the convert_model entry style and the EXPECT_THROW the
//   fix should produce once length is validated against the mapped size.

#include "onnx_utils.hpp"            // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/test_constants.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, mmap_length_absent_must_clamp_to_mapped_size) {
    // TODO: provide a crafted .onnx + companion external-data file where:
    //   - external_data has only {location, offset=0}, NO 'length' (m_data_length==0)
    //   - the external-data file is shorter than the size implied at stat time
    // Pre-fix: load_external_mmap_data builds a SharedBuffer of (file_size - m_offset)
    //   bytes over a smaller mapping -> ASan over-read on first tensor access.
    // Post-fix: length is computed from mapped_memory->size() - m_offset and the
    //   offset is bounds-checked, so the malformed/short file is rejected.
    //
    // EXPECT_THROW(convert_model("external_data/external_data_length_absent_short.onnx"),
    //              ov::Exception);
    GTEST_SKIP() << "TODO: needs crafted .onnx + short external-data fixture and a "
                    "TOCTOU-shrink injection between file_size() and load_mmap_object().";
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.mmap_length_absent_must_clamp_to_mapped_size . Expected pre-fix failure: AddressSanitizer reports a heap/mmap over-read when accessing the SharedBuffer beyond mapped_memory->size(); post-fix the crafted short-file model is rejected via ov::Exception (invalid_external_data). TODO fixtures: external_data/external_data_length_absent_short.onnx plus a companion data file shorter than the stat-time size, and a hook to shrink the file between tensor_external_data.cpp:52 and :62.

## Suggested fix
After the corrected post-mmap check (see finding 1), compute the fallback length from the mapped size, not from the stale file_size:

  const uint64_t effective_length = (m_data_length > 0)
      ? m_data_length
      : mapped_memory->size() - m_offset;  // use actual mapping size
  return std::make_shared<ov::SharedBuffer<...>>(
      mapped_memory->data() + m_offset, effective_length, mapped_memory);

This eliminates reliance on the pre-mmap stat value entirely for the length calculation.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #401.
