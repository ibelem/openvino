# Security finding #449: When `ov::util::file_size` returns −1 (documented at file_util.hpp …

**Summary:** When `ov::util::file_size` returns −1 (documented at file_util.hpp …

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error / CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** In External_Stream memory mode, `allocate_data(SIZE_MAX)` executes `new uint8_t[SIZE_MAX]` (graph_iterator_proto.hpp line 136), triggering `std::bad_alloc` or exhausting heap memory. This is a denial-of-service against any application that loads an attacker-supplied ONNX model with externally stored tensor data (e.g., model serving, ONNX model import in OpenVINO). In External_MMAP mode, if `load_mmap_object` somehow succeeds, `m_tensor_data_size = SIZE_MAX` leads to downstream OOB reads when consumers iterate over the tensor data. Secondary runtime mitigations (ifstream fail-check at line 347 for a nonexistent file; `load_mmap_object` throwing for a nonexistent file) partially block exploitation but only when `file_size == -1` is caused by the file being absent—not when it is caused by stat failures on accessible special files (proc, sysfs, network FS) or under TOCTOU race.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:303` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled `location`, `offset`, and `length` fields in the ONNX external-data key-value pairs, parsed from an untrusted ONNX model file at lines 283–292.

## Description / Root cause
When `ov::util::file_size` returns −1 (documented at file_util.hpp line 154: `return ec ? -1 : static_cast<int64_t>(size)`) and the attacker supplies `ext_data_length == 0`, all three branches of the compound guard at lines 304–305 evaluate to false: (1) `file_size <= 0 && ext_data_length > 0` → `true && false` → false; (2) `ext_data_length > static_cast<uint64_t>(file_size)` → `0 > UINT64_MAX` → false (signed −1 promotes to UINT64_MAX); (3) `ext_data_offset > static_cast<uint64_t>(file_size) − ext_data_length` → `ext_data_offset > UINT64_MAX − 0` → always false. The guard is therefore entirely bypassed. Control reaches line 312–314, where `resolved_data_length = static_cast<size_t>(file_size) − static_cast<size_t>(ext_data_offset)` = `SIZE_MAX − 0` = `SIZE_MAX`. This value is assigned to `tensor_meta_info.m_tensor_data_size` at line 352 and passed to `allocate_data(SIZE_MAX)` at line 353.

**Validator analysis:** The signed-to-unsigned conversion is real. file_size() returns int64_t -1 on stat failure (file_util.hpp:151-154). With ext_data_length==0 (length key absent — a legitimate ONNX 'rest of file' case), all three disjuncts of the guard at 304-305 are false: (1) requires ext_data_length>0; (2) 0 > (uint64_t)(-1) is false; (3) 0 > UINT64_MAX-0 is false. Control reaches line 312-314 where the else-branch computes static_cast<size_t>(file_size) - offset = SIZE_MAX. CWE-195/CWE-789 categorisation is accurate and the proposed fix (reject file_size<0 before any cast, then validate offset/length against the unsigned size) is correct and sufficient. Two caveats on impact: (a) `new uint8_t[SIZE_MAX]` throws std::bad_array_new_length/length_error immediately rather than actually exhausting the heap, so the External_Stream path is an uncaught-exception DoS, not real memory exhaustion; (b) the more serious consequence is External_MMAP mode (line 329) where m_tensor_data_size=SIZE_MAX with m_tensor_data=mapped_memory->data()+offset enables downstream OOB reads if load_mmap_object succeeds on a non-regular file. Reachability hinges on file_size==-1 while the file is still openable (special/proc files, network FS, TOCTOU) — plausible from untrusted convert_model on a crafted .onnx, since for plain-missing files the stream fail-check (347) and load_mmap_object both throw. Valid but narrow.

## Exploit / Proof of Concept
Craft a `.onnx` file with an externally stored tensor whose `external_data` has: `location = 'weights.bin'` (a path that exists but whose `std::filesystem::file_size` returns an error, e.g. a symlink to a `/proc` pseudo-file on Linux), `offset = '0'`, and `length` key absent (defaults to 0). When parsed at lines 283–292, `ext_data_length = 0`. `ov::util::file_size(full_path)` returns −1. The guard at lines 304–305 is entirely bypassed as shown above. Line 314 computes `resolved_data_length = SIZE_MAX`. In External_Stream mode the check at line 347 may pass if the underlying fd is still openable. `allocate_data(SIZE_MAX)` (line 353) → `new uint8_t[18446744073709551615]` → `std::bad_alloc` propagates uncaught through `extract_tensor_external_data` → `extract_tensor_meta_info` → caller, crashing or aborting the inference engine process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for graph_iterator_proto.cpp:304-314: a negative file_size
// (ov::util::file_size returns -1, file_util.hpp:154) combined with an absent
// 'length' (ext_data_length==0) must be REJECTED, not silently turned into
// resolved_data_length = SIZE_MAX -> allocate_data(SIZE_MAX) at line 353.
// Pre-fix: convert_model triggers std::bad_array_new_length / OOB and may not
// throw ov::Exception cleanly. Post-fix: a guarded runtime_error is raised.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// gtest filter: --gtest_filter=*external_data_negative_filesize*

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_import_external_data, external_data_negative_filesize_zero_length) {
    // TODO: provide a crafted fixture under models/external_data/ whose tensor's
    // external_data sets location to a path for which std::filesystem::file_size
    // returns an error (ec != 0) while the path is still openable as a stream —
    // e.g. a non-regular/special file or a fixture installed at test time as a
    // FIFO/symlink. A plain missing file is NOT sufficient because the ifstream
    // fail-check at line 347 (or load_mmap_object) would throw for a different
    // reason. The 'length' key must be ABSENT so ext_data_length defaults to 0.
    //
    // The assertion the regression encodes: the model load must fail with a
    // controlled ov::Exception/runtime_error, and must NOT reach
    // allocate_data(SIZE_MAX) (no std::bad_array_new_length / ASan abort).
    OV_EXPECT_THROW(convert_model("external_data/external_data_negative_filesize.onnx"),
                    ov::Exception,
                    testing::HasSubstr("external"));
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests. Run: ./ov_onnx_frontend_tests --gtest_filter=*external_data_negative_filesize*. Pre-fix expectation (with ASan): SUMMARY: AddressSanitizer / std::bad_array_new_length thrown from allocate_data (new uint8_t[SIZE_MAX]) at graph_iterator_proto.hpp:136, or an OOB read in External_MMAP mode; test fails because no clean ov::Exception is produced. Post-fix: the added `if (file_size < 0) throw` guard makes convert_model raise a controlled exception and the test passes. NOTE: requires a crafted .onnx fixture plus a special-file condition forcing ov::util::file_size to return -1 while still openable — hence skeleton.

## Suggested fix
Replace the compound guard with a check that explicitly rejects any negative `file_size` before any cast, and separately validates `ext_data_offset` and `ext_data_length`:
```cpp
if (file_size < 0) {
    throw std::runtime_error("Cannot determine size of external data file: " + ext_location);
}
const uint64_t fsize = static_cast<uint64_t>(file_size);
if (ext_data_offset > fsize) {
    throw std::runtime_error("ext_data_offset exceeds file size");
}
if (ext_data_length > 0 && ext_data_length > fsize - ext_data_offset) {
    throw std::runtime_error("ext_data_length exceeds available file bytes");
}
```
Then compute `resolved_data_length` safely:
```cpp
const size_t resolved_data_length = (ext_data_length > 0)
    ? static_cast<size_t>(ext_data_length)
    : static_cast<size_t>(fsize - ext_data_offset);
```
This eliminates the signed-to-unsigned promotion path entirely.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #449.
