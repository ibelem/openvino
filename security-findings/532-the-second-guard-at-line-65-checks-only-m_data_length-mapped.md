# Security finding #532: The second guard at line 65 checks only `m_data_length > mapped_mem…

**Summary:** The second guard at line 65 checks only `m_data_length > mapped_mem…

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-367: TOCTOU Race Condition
**Severity / Impact:** An attacker supplying a malicious ONNX model can trigger a read up to `m_offset` bytes (up to 2^64–1 from the protobuf field) past the end of the memory-mapped region. On Linux/Windows this crashes the process (SIGSEGV / access violation) achieving denial of service; under certain ASLR and process layouts it may also leak adjacent mapped memory contents (weights of other models, heap metadata, etc.) via the returned `SharedBuffer`.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX protobuf external-data fields: `m_offset` (uint64_t) and `m_data_length` (uint64_t) are parsed without upper-bound caps via `std::stoull` at lines 24 and 26 from an attacker-supplied model file.

## Description / Root cause
The second guard at line 65 checks only `m_data_length > mapped_memory->size()`, never the combined `m_offset + m_data_length > mapped_memory->size()`. The first guard (lines 53–54) validates `m_offset + m_data_length <= file_size` where `file_size` is captured at line 52 from `ov::util::file_size()`. However, `mapped_memory->size()` (the actual mapped region) can legally differ from `file_size` via two paths: (A) TOCTOU — the file is shrunk on-disk between the `file_size()` call at line 52 and `load_mmap_object()` at line 62, so the mmap reflects the smaller truncated file; (B) stale cache — a prior tensor call stored a `MappedMemory` for the same path (lines 57–63 cache on `full_path`), the file was later grown, and a new tensor's first guard now validates against the larger `file_size` while the cache returns the smaller mapping. In both cases, the second guard at line 65 passes (`m_data_length <= mapped_memory->size()` is satisfied with a small `m_data_length`), yet the pointer arithmetic at line 69 (`mapped_memory->data() + m_offset`) walks `m_offset` bytes past the start of a region that is smaller than the validated `file_size`, producing an out-of-bounds read.

**Validator analysis:** The vuln type (CWE-125 OOB Read + CWE-367 TOCTOU) is accurate. The first guard (lines 53–54) uses `file_size` from `ov::util::file_size()` as the bound; the second guard (line 65) uses `mapped_memory->size()`, which can legally be smaller if (A) the file is truncated between the `file_size()` call (line 52) and `load_mmap_object()` call (line 62), or (B) the cache (lines 57–63) returns a mapping created when the file was smaller. In both cases `m_offset` is never checked against `mapped_memory->size()`, so line 69 (`mapped_memory->data() + m_offset`) reads past the end of the mapped region. Path B (stale cache) is entirely deterministic — two tensors referencing the same file with different offsets, interleaved with a file-growth between loads — and requires no external race. The proposed fix is correct and sufficient: replacing the line-65 guard with `m_data_length > mapped_memory->size() || m_offset > mapped_memory->size() - m_data_length` makes `mapped_memory->size()` the authoritative bound. The additional suggestion to invalidate cache entries when sizes diverge is a sound defence-in-depth measure but not strictly required if the combined guard is in place.

## Exploit / Proof of Concept
Path A (pure TOCTOU, no prior cache): Supply a model with `location=weights.bin`, `offset=990000`, `length=9000`. When `load_external_mmap_data` is called, `file_size()` at line 52 reads 1 000 000 bytes — the first guard passes (990000+9000 == 1000000 <= 1000000). The attacker truncates `weights.bin` to 100 000 bytes before `load_mmap_object()` executes at line 62. The mmap succeeds over the now-100 KB file; `mapped_memory->size()` == 100 000. The second guard at line 65 sees `9000 > 100000` = false, passes. `mapped_memory->data() + 990000` (line 69) reads 890 000 bytes past the end of the mapped region. — Path B (stale cache): Include two tensors referencing the same `location`. Tensor T1 uses `offset=0, length=100000` with the file at 100 KB; the 100 KB `MappedMemory` is inserted into `cache`. The attacker then grows the file to 1 MB on disk. Tensor T2 uses `offset=990000, length=9000`; `file_size()` returns 1 MB, first guard passes, cache returns the stale 100 KB mapping, second guard passes, OOB read occurs at line 69.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for tensor_external_data.cpp:65-69:
// The stale-cache path allows m_offset to be unchecked against mapped_memory->size(),
// causing an OOB read at line 69 (`mapped_memory->data() + m_offset`).
// Pre-fix: this test triggers a heap-buffer-overflow (detected by ASan) or silent OOB.
// Post-fix: load_external_mmap_data throws ov::Exception (invalid_external_data).

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

// Internal header — adjust include path to match build-tree layout
#include "utils/tensor_external_data.hpp"
#include "exceptions.hpp"           // error::invalid_external_data
#include "openvino/util/mmap_object.hpp"

namespace ov_onnx_ext_data_test {

// Write `size` bytes of value 0xAB to a temp file; return the path.
static std::filesystem::path write_tmp_file(const std::string& name, std::size_t size) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(size, static_cast<char>(0xAB));
    f.write(buf.data(), static_cast<std::streamsize>(size));
    return path;
}

} // namespace ov_onnx_ext_data_test

using namespace ov_onnx_ext_data_test;

// Validates that a stale cache entry (mapped_memory->size() < file_size)
// cannot be exploited via an unchecked m_offset.
//
// Setup:
//   1. Write a 100-byte file; mmap it; insert into cache (size = 100 B).
//   2. Grow the file to 1000 B on disk.
//   3. Construct TensorExternalData with offset=900, length=9.
//      - First guard: 900+9 <= file_size(1000) → passes.
//      - Second guard pre-fix: 9 > 100 → false → passes; line 69 OOBs by ~809 bytes.
//      - Second guard post-fix: 900 > 100-9 → true → throws.
TEST(OnnxExternalDataMmap, StaleCache_LargeOffsetWithSmallLength_ThrowsOrASan) {
    const std::string filename = "ov_ext_data_stale_cache.bin";
    auto file_path = write_tmp_file(filename, 100);
    auto model_dir  = file_path.parent_path();

    // Pre-populate cache with the 100-B mapping.
    auto small_mapping = ov::load_mmap_object(file_path);
    ASSERT_EQ(small_mapping->size(), std::size_t{100})
        << "mmap of 100-B file must report size 100";

    auto cache = std::make_shared<
        std::map<std::filesystem::path, std::shared_ptr<ov::MappedMemory>>>();
    (*cache)[file_path] = small_mapping;

    // Grow the file to 1000 B so that ov::util::file_size() now returns 1000
    // while the cache still holds the 100-B MappedMemory.
    {
        std::ofstream f(file_path, std::ios::binary | std::ios::trunc);
        std::vector<char> buf(1000, static_cast<char>(0xCD));
        f.write(buf.data(), 1000);
    }

    // offset=900, length=9: valid against file_size(1000), OOB against mapped size(100).
    ov::frontend::onnx::detail::TensorExternalData ext_data(
        filename, /*offset=*/900, /*length=*/9);

    // Post-fix: must throw; pre-fix: ASan heap-buffer-overflow on line 69.
    EXPECT_THROW(
        { ext_data.load_external_mmap_data(model_dir, cache); },
        ov::Exception);  // error::invalid_external_data derives from ov::Exception

    std::filesystem::remove(file_path);
}

// Validate the boundary case: offset exactly at the end of the mapped region.
TEST(OnnxExternalDataMmap, StaleCache_OffsetAtMappedEnd_ThrowsOrASan) {
    const std::string filename = "ov_ext_data_boundary.bin";
    auto file_path = write_tmp_file(filename, 50);
    auto model_dir  = file_path.parent_path();

    auto small_mapping = ov::load_mmap_object(file_path);
    ASSERT_EQ(small_mapping->size(), std::size_t{50});
    auto cache = std::make_shared<
        std::map<std::filesystem::path, std::shared_ptr<ov::MappedMemory>>>();
    (*cache)[file_path] = small_mapping;

    // Grow to 500 B.
    {
        std::ofstream f(file_path, std::ios::binary | std::ios::trunc);
        std::vector<char> buf(500, static_cast<char>(0xEF));
        f.write(buf.data(), 500);
    }

    // offset=50 == mapped size, length=1 → OOB by 1 byte at line 69.
    ov::frontend::onnx::detail::TensorExternalData ext_data(
        filename, /*offset=*/50, /*length=*/1);

    EXPECT_THROW(
        { ext_data.load_external_mmap_data(model_dir, cache); },
        ov::Exception);

    std::filesystem::remove(file_path);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (with -fsanitize=address,undefined). Filter: --gtest_filter='OnnxExternalDataMmap.*'. Expected pre-fix failure: AddressSanitizer heap-buffer-overflow on tensor_external_data.cpp:69 ('mapped_memory->data() + m_offset' reads past the end of the mapped region). Expected post-fix result: both tests PASS with no sanitizer errors.

## Suggested fix
Replace the second guard at line 65 with a check that mirrors the first guard but uses `mapped_memory->size()` as the authoritative bound:
```cpp
if (mapped_memory->size() == 0 ||
    m_data_length > mapped_memory->size() ||
    m_offset > mapped_memory->size() - m_data_length) {
    throw error::invalid_external_data{*this};
}
```
This ensures `m_offset + m_data_length <= mapped_memory->size()` regardless of any discrepancy between `file_size` and the actual mapped region size caused by TOCTOU or stale cache. Additionally, consider either (a) invalidating the cache entry whenever `mapped_memory->size() != static_cast<size_t>(file_size)` so stale entries cannot persist, or (b) storing the file size alongside the cache entry and re-mapping when sizes diverge.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #532.
