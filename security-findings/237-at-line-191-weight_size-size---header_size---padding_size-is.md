# Security finding #237: At line 191, `weight_size = size - header_size - padding_size` is c…

**Summary:** At line 191, `weight_size = size - header_size - padding_size` is c…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error / CWE-681: Incorrect Conversion between Numeric Types
**Severity / Impact:** If the implicitly-converted negative seek value is small in magnitude (e.g., -25), the seek lands within the valid file region, `s.good()` returns true, and the function returns true. The outer TLV scanner then continues parsing from that backward position, re-reading and re-interpreting already-processed file regions as fresh TLV records — a record-injection / reparse attack. This can corrupt in-memory cache index state (m_blob_index, m_weight_registry, m_cache_sources) and may lead to subsequent OOB reads or use of stale/attacker-injected metadata when loading compiled blobs.
**Affected location:** `targets/openvino/src/inference/src/single_file_storage.cpp:191` — `SingleFileStorage::build_content_index (weight_source_reader lambda)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted on-disk cache file parsed in build_content_index; the TLV `size` field of a WeightSource record is attacker-controlled.

## Description / Root cause
At line 191, `weight_size = size - header_size - padding_size` is computed as an unsigned `TLVTraits::LengthType` (uint64_t). At line 193, this value is passed to `s.seekg(weight_size, std::ios::cur)` without any explicit cast or upper-bound check against `std::numeric_limits<std::streamoff>::max()`. `std::streamoff` is a signed type (typically int64_t). When `weight_size > INT64_MAX`, the implicit narrowing conversion wraps the value to a large negative `std::streamoff`, issuing a backward seek instead of a forward one. Contrast with `blob_reader` (line 112), which at least does `static_cast<std::streamoff>(...)` explicitly, making the narrowing visible. `weight_source_reader` has no cast at all.

**Validator analysis:** The flaw is real and reachable. In scan_tlv_records (dev/tlv_format.cpp:92) the bound check `stream_end - tellg() < static_cast<std::streamoff>(size)` casts size to a SIGNED std::streamoff: for size=UINT64_MAX this becomes -1, so `positive < -1` is false and the oversized record is NOT rejected — size=UINT64_MAX is passed straight to weight_source_reader. There, header_size=24, the `padding_size > size-header_size` check (line 184) passes with padding=0, weight_size = UINT64_MAX-24 = 0xFFFFFFFFFFFFFFE7, and `s.seekg(weight_size, std::ios::cur)` (line 193) implicitly narrows that uint64 to streamoff = -25, performing a backward seek that still leaves s.good()==true. The function returns true and scan_tlv_records re-parses bytes already consumed — at minimum an infinite-loop DoS, and plausibly record-injection corrupting m_blob_index/m_weight_registry/m_cache_sources. CWE-195/CWE-681 is the correct categorization. The proposed fix (reject `weight_size > static_cast<uint64_t>(numeric_limits<std::streamoff>::max())` then explicit cast) is correct and sufficient for line 191-193, BUT it is incomplete: the same signed-narrowing bug exists in the upstream guard scan_tlv_records (tlv_format.cpp:92) and in blob_reader (line 112 casts without an upper-bound check, so a backward seek of arbitrary magnitude is also possible there). A complete fix must (a) add the upper-bound check before EVERY seekg(size/derived, cur) including blob_reader, and ideally (b) fix scan_tlv_records line 92 to compare in unsigned space (e.g. `static_cast<uint64_t>(stream_end - tellg()) < size`) so oversized records are rejected at the scanner rather than relying on each reader.

## Exploit / Proof of Concept
Craft a WeightSource TLV record with `size = UINT64_MAX` and `padding_size = 0`. After reading the 24-byte header (device_id + source_id + padding_size), the check `padding_size (0) > size (UINT64_MAX) - header_size (24)` is false (passes). Then `weight_size = UINT64_MAX - 24 = 0xFFFFFFFFFFFFFFE7`. Implicit conversion to `std::streamoff` (int64_t) yields `-25`. `s.seekg(-25, std::ios::cur)` succeeds if the current position is >= 25 bytes from file start (which it always is after parsing a header and prior records). `s.good()` returns true, the function returns true, and the TLV scanner re-parses 25 bytes back in the file.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-195/CWE-681 at
//   openvino/src/inference/src/single_file_storage.cpp:191-193 (weight_source_reader)
//   and the upstream guard openvino/src/inference/src/dev/tlv_format.cpp:92.
// A WeightSource TLV record with size=UINT64_MAX, padding_size=0 makes
//   weight_size = UINT64_MAX-24 = 0xFFFFFFFFFFFFFFE7, which narrows to a
//   negative std::streamoff (-25) -> backward seek -> reparse / DoS.
// Pre-fix: build_content_index either loops/reparses or accepts the crafted
//   file (returns success). Post-fix: the oversized record is rejected.
//
// NOTE: build_content_index() is private and there is no public accessor that
// returns its boolean directly, so this is a SKELETON: it crafts the on-disk
// cache file and drives the public read path. The exact inference-component
// test target and public entry point must be confirmed against the real test
// tree before use.
#include <gtest/gtest.h>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>
#include "openvino/runtime/single_file_storage.hpp"
#include "openvino/runtime/tlv_format.hpp"

using namespace ov::runtime;

namespace {
// TODO: confirm the on-disk header layout (version: 3x uint16) and that
// initialize()/get_context() actually invokes build_content_index on an
// existing file. Symbol names below are taken from
// dev_api/openvino/runtime/single_file_storage.hpp.
void write_u16(std::ostream& s, uint16_t v) { s.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_u32(std::ostream& s, uint32_t v) { s.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_u64(std::ostream& s, uint64_t v) { s.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
}

TEST(SingleFileStorageSecurity, WeightSourceOversizedSizeIsRejected) {
    const std::filesystem::path path = "crafted_weightsource_cache.bin";
    {
        std::ofstream f(path, std::ios::binary);
        // version {0,1,0} matching SingleFileStorage::m_version
        write_u16(f, 0); write_u16(f, 1); write_u16(f, 0);
        // TLV record: tag = WeightSource (0x11), size = UINT64_MAX
        write_u32(f, static_cast<uint32_t>(SingleFileStorage::Tag::WeightSource));
        write_u64(f, std::numeric_limits<uint64_t>::max());
        // WeightSource value header: device_id, source_id, padding_size(=0)
        write_u64(f, 1);   // device_id
        write_u64(f, 7);   // source_id
        write_u64(f, 0);   // padding_size
        // a few trailing bytes so a backward seek lands in-bounds
        std::vector<char> pad(64, 0);
        f.write(pad.data(), pad.size());
    }

    // The crafted file must NOT cause an infinite reparse loop and must NOT be
    // silently accepted as a valid index. Post-fix the parser rejects it.
    SingleFileStorage storage(path);
    // TODO: replace with the real public entry point that triggers
    // build_content_index (e.g. initialize() then a read). Assert it does not
    // hang and surfaces failure rather than re-parsing already-consumed bytes.
    EXPECT_NO_FATAL_FAILURE(storage.initialize());
}
```
**Build / run:** Build target: the inference component's gtest target (e.g. ov_inference_unit_tests / ov_dev_api_tests — confirm from src/inference/tests). Run: ov_inference_unit_tests --gtest_filter=SingleFileStorageSecurity.WeightSourceOversizedSizeIsRejected with ASan enabled. Pre-fix expectation: test hangs (infinite reparse from the -25 backward seek) or accepts the crafted index; ASan may report later OOB use of stale m_cache_sources/m_blob_index metadata. Post-fix: the oversized WeightSource record is rejected (build_content_index returns false / parse fails) and the test passes cleanly.

## Suggested fix
Add an explicit upper-bound check before the seek, mirroring the pattern in `blob_reader` but with the additional range assertion: `if (weight_size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;`. Then cast explicitly: `s.seekg(static_cast<std::streamoff>(weight_size), std::ios::cur);`. This mirrors the blob_reader pattern at line 112 and makes the narrowing visible and safe.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #237.
