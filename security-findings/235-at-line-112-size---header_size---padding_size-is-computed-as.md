# Security finding #235: At line 112, `size - header_size - padding_size` is computed as `ui…

**Summary:** At line 112, `size - header_size - padding_size` is computed as `ui…

**CWE IDs:** CWE-681: Incorrect Conversion between Numeric Types / CWE-125: Out-of-bounds Read
**Severity / Impact:** Downstream in `read_cache_entry` (line 259), the stored `blob_size` (~UINT64_MAX) is passed as the shape dimension to `read_tensor_data`. This causes either (a) an enormous memory-mapping / allocation request that crashes the process (DoS / CWE-789), or (b) an out-of-bounds read of almost the entire virtual address space if the allocation succeeds. Any user or process loading a maliciously crafted cache file is affected.
**Affected location:** `targets/openvino/src/inference/src/single_file_storage.cpp:112` — `SingleFileStorage::build_content_index (blob_reader lambda)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted on-disk cache file → TLV record length field (uint64_t) read by build_content_index

## Description / Root cause
At line 112, `size - header_size - padding_size` is computed as `uint64_t` (confirmed: TLVTraits::LengthType = uint64_t, PadSizeType = uint64_t). The line-105 check `padding_size > size - header_size` prevents unsigned underflow, but places NO upper bound ensuring the result fits in a signed 64-bit integer. If the attacker supplies `size` such that `size - header_size - padding_size > INT64_MAX` (e.g. `size = 0xFFFFFFFFFFFFFF10`, `padding_size = 0`, giving residual `0xFFFFFFFFFFFFFF00 = -256` as streamoff), the `static_cast<std::streamoff>` at line 112 produces a negative value. That negative value: (a) is used as a backward seek offset at line 113 — which can succeed (stream stays `good()`) if the resulting position is still inside the file; and (b) is stored at line 118 via `static_cast<uint64_t>(blob_data_size)` → a value near UINT64_MAX (~2^64 - 256) into `m_blob_index[id].size`.

**Validator analysis:** Confirmed: TLVTraits::LengthType and PadSizeType are uint64_t (tlv_format.hpp:27, single_file_storage.hpp:66). The line-105 guard prevents unsigned underflow of size-header_size but imposes no INT64_MAX upper bound, so with size>INT64_MAX the line-112 static_cast<std::streamoff> yields a negative offset, and line 118's static_cast<uint64_t>(negative) stores ~UINT64_MAX into m_blob_index[id].size. The would-be upstream mitigation in scan_tlv_records (dev/tlv_format.cpp:92) is itself broken: it compares against static_cast<std::streamoff>(size), which is negative for size>INT64_MAX, so the 'remaining < size' check is bypassed and the oversized record reaches the reader lambda. CWE-681 (incorrect numeric conversion) is accurate; the corruption is real. Reachability requires the negative seekg(-256) to keep the stream good() (position must be >=256 into the file, achievable with preceding crafted records) AND build_content_index to ultimately return true so initialize() does not throw at OPENVINO_ASSERT (single_file_storage.cpp:346) — engineerable since the attacker controls the full file, though it requires careful layout. The downstream impact wording is slightly optimistic: at read_cache_entry:259 the huge uint64 size is cast to PartialShape::value_type (signed int64), which for the 0xFF..F00 case becomes -256, i.e. an invalid/negative dimension into read_tensor_data — still a corruption/DoS, but the precise '16 EiB mmap' claim depends on the chosen size value. The proposed fix is correct and sufficient: bound the residual to std::numeric_limits<std::streamoff>::max() before the cast at line 112, and apply the identical guard to weight_source_reader at line 191. Additionally the root upstream defect should be fixed in scan_tlv_records:92 by comparing as unsigned (e.g. static_cast<uint64_t>(stream_end - tellg) < size) so oversized records are rejected before reaching any reader.

## Exploit / Proof of Concept
Craft a cache file whose TLV Blob record has `size = 0xFFFFFFFFFFFFFF10` (tag + length read from file), `id` = any valid uint64, `padding_size = 0`. `size - header_size(16) - padding_size(0) = 0xFFFFFFFFFFFFFF00`. `static_cast<streamoff>` yields -256. `s.seekg(-256, ios::cur)` from a position ≥ 256 (e.g. after the version header + tag + length fields) stays within the file so `s.good()` is true, line 114 check passes. Line 118 stores `static_cast<uint64_t>(-256) = 18446744073709551360` as `m_blob_index[id].size`. When `read_cache_entry` is later called with `enable_mmap=true`, line 259 calls `read_tensor_data(path, u8, {18446744073709551360}, blob_pos)`, triggering a ~16 EiB mmap / allocation and OOB read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-681 at single_file_storage.cpp:112 / :118 (and :191).
// Pre-fix: a crafted cache file whose Blob TLV length field > INT64_MAX makes
//   static_cast<std::streamoff>(size - header - pad) negative (line 112), which
//   is then stored as ~UINT64_MAX via static_cast<uint64_t> at line 118, and
//   propagated to read_cache_entry:259 as an invalid shape dimension.
// Post-fix: build_content_index must reject the oversized residual, so loading
//   the cache must NOT yield a usable blob entry / must throw on corrupt input.
//
// TODO: confirm the ov_*_tests target and existing test file for src/inference
//       (e.g. the inference unit-test tree) and the SingleFileStorage include path.
// TODO: SingleFileStorage::m_blob_index/build_content_index are private; this test
//       drives the public surface (ctor -> initialize() -> read_cache_entry()).
#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "openvino/runtime/single_file_storage.hpp"  // dev_api

using ov::runtime::SingleFileStorage;

namespace {
template <typename T>
void put(std::vector<char>& b, T v) {
    const char* p = reinterpret_cast<const char*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
}  // namespace

TEST(SingleFileStorageSecurity, BlobLengthOverflowRejected) {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "ov_sfs_overflow_repro.bin";

    std::vector<char> buf;
    // Version header (major, minor, patch) matching SingleFileStorage::m_version {0,1,0}.
    put<uint16_t>(buf, 0);
    put<uint16_t>(buf, 1);
    put<uint16_t>(buf, 0);

    // TODO: a single backward-seek Blob record alone makes build_content_index loop/
    //       re-parse; a faithful PoC needs a preceding skipped record (>=256 bytes)
    //       so seekg(-256) lands inside the file and the scan still terminates true.
    //       Fill that in to exercise the full :118 -> :259 propagation.
    // Malicious Blob TLV: tag = Tag::Blob (0x03, uint32), length = 0xFFFFFFFFFFFFFF10.
    put<uint32_t>(buf, static_cast<uint32_t>(SingleFileStorage::Tag::Blob));
    put<uint64_t>(buf, 0xFFFFFFFFFFFFFF10ULL);  // length > INT64_MAX -> negative streamoff at line 112
    put<uint64_t>(buf, /*BlobId*/ 1ULL);
    put<uint64_t>(buf, /*PadSize*/ 0ULL);

    {
        std::ofstream os(path, std::ios::binary);
        os.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }

    SingleFileStorage storage(path);
    // Pre-fix: initialize() succeeds and m_blob_index[1].size == ~UINT64_MAX, then
    //   read_cache_entry triggers an invalid shape dimension in read_tensor_data.
    // Post-fix: the oversized residual is rejected, so loading the cache throws
    //   (OPENVINO_ASSERT at single_file_storage.cpp:346) instead of poisoning the index.
    EXPECT_THROW(
        {
            storage.initialize();
            storage.read_cache_entry("1", /*enable_mmap=*/true,
                                     [](auto&) {});
        },
        ov::Exception);

    std::error_code ec;
    fs::remove(path, ec);
}
```
**Build / run:** Build: cmake --build . --target ov_inference_unit_tests (confirm exact target from src/inference/tests). Run: ./ov_inference_unit_tests --gtest_filter=SingleFileStorageSecurity.BlobLengthOverflowRejected . Pre-fix expectation: with ASan, the poisoned m_blob_index size (~UINT64_MAX) propagates to read_tensor_data at single_file_storage.cpp:259 yielding an invalid/negative shape dimension (heap-buffer-overflow / bad-alloc / abort) and the EXPECT_THROW(ov::Exception) fails because no proper exception is raised at parse time. Post-fix: build_content_index rejects the oversized residual and loading throws ov::Exception, test passes. TODO: adjust target name and add the >=256-byte preceding record needed to keep the stream good() through the backward seek.

## Suggested fix
Before the cast at line 112, verify the computed residual fits in a signed 64-bit value. Add: `const uint64_t raw_data_size = size - header_size - padding_size; if (raw_data_size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false; const auto blob_data_size = static_cast<std::streamoff>(raw_data_size);`. This replaces lines 112 and makes the cast safe. Similarly apply the same guard to `weight_source_reader` at line 191-193 which has the identical pattern.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #235.
