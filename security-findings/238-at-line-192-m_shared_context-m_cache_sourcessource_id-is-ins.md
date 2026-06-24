# Security finding #238: At line 192, `m_shared_context->m_cache_sources[source_id] = {}` is…

**Summary:** At line 192, `m_shared_context->m_cache_sources[source_id] = {}` is…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Two distinct impacts: (1) Any malformed WeightSource record permanently inserts an empty `{}` entry for the attacker-controlled `source_id` into `m_cache_sources`, polluting the cache context even when parsing should fail. (2) When combined with the backward-seek exploit, the function returns `true`, causing the TLV scanner to continue from the rewound stream position and parse earlier file content as new records, injecting attacker-chosen metadata into m_blob_index or m_weight_registry. Both lead to incorrect or malicious in-memory state used in subsequent blob loads.
**Affected location:** `targets/openvino/src/inference/src/single_file_storage.cpp:192` — `SingleFileStorage::build_content_index (weight_source_reader lambda)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted on-disk cache file parsed in build_content_index; `source_id` is read from the file without validation.

## Description / Root cause
At line 192, `m_shared_context->m_cache_sources[source_id] = {}` is inserted unconditionally into the shared-context map BEFORE the `s.seekg(weight_size, std::ios::cur)` and `s.good()` checks at lines 193-194. If the record is malformed (e.g., bad seek) the empty entry persists in the map. Worse, when the backward-seek exploit (Finding 1) succeeds — `s.good()` is true — the function returns true with both the map poisoned and the stream position rewound. Compare with `blob_reader` (lines 117-118), which only writes to `m_blob_index[id]` after the full validity check completes.

**Validator analysis:** Confirmed real defect: in weight_source_reader the map entry is inserted (line 192) before the final seek/good validation (193-194), inconsistent with blob_reader's post-validation write (113-118). This is a genuine CWE-20/ordering bug: on a malformed WeightSource record where the trailing seekg fails, the function returns false but the empty {} entry for the attacker-controlled source_id already persists in the shared m_cache_sources map (impact 1, accurate). Impact 2 (returns true with rewound stream) is contingent on Finding-1's backward-seek mechanism: weight_size = size-header-padding is unsigned (LengthType); a huge `size` makes weight_size wrap and, cast to signed std::streamoff in seekg, becomes a negative/backward seek that can leave s.good()==true, so the function returns true with a rewound position and a poisoned map — plausible but it depends on the separate Finding-1 flaw. vulnType (Improper Input Validation) is accurate; impact-1 is concrete, impact-2 is real only in combination. The proposed fix (move the assignment after the seek+good() check, matching blob_reader) is correct and SUFFICIENT for impact-1, but INSUFFICIENT for impact-2: even post-move, a backward/over-long seek that keeps good()==true would still insert the entry and return true. The fix should additionally reject the record when weight_size would seek past the stream bounds or produce a negative cumulative offset (validate weight_size <= remaining stream length and use unsigned-safe bounds checks before seekg), as the better remedy. The EP repo is not reachable on the cited boundary, hence rejected there.

## Exploit / Proof of Concept
Trigger the backward-seek scenario from Finding 1 (size=UINT64_MAX, padding_size=0). Line 192 writes `m_cache_sources[source_id] = {}` with an attacker-chosen `source_id`. Then line 193 issues seekg(-25) which succeeds, line 194 returns true. The outer `scan_tlv_records` loop advances its position counter by `UINT64_MAX` (the original TLV size field) but the actual stream is 25 bytes behind where it should be, leading to the scanner re-processing earlier file content as a new TLV record. If that earlier content forms a valid-looking Blob or ConstantMeta record, m_blob_index or m_weight_registry are further corrupted.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/inference/src/single_file_storage.cpp:192-194.
// Pre-fix: a WeightSource record whose trailing seekg fails (or wraps) still
// leaves an empty m_cache_sources[source_id] entry AND/OR returns true with a
// rewound stream. Post-fix: the map MUST NOT contain the attacker source_id
// when build_content_index reports failure, and a malformed/over-long size MUST
// cause the record (and overall scan) to be rejected.
//
// SKELETON: SingleFileStorage is an internal inference component and the trigger
// requires a crafted on-disk TLV cache blob, so exact symbols/fixtures must be
// confirmed against the inference component's own test tree before use.

#include <gtest/gtest.h>
#include <fstream>
#include <cstdint>
#include <string>

// TODO: replace with the real header that declares SingleFileStorage and the
// shared cache context (grep openvino/src/inference/src for the class def).
// #include "single_file_storage.hpp"

namespace {

// TODO: confirm Tag::WeightSource value and TLV record layout:
//   [tag:TagType][len:LengthType][device_id:DataIdType][source_id:DataIdType]
//   [padding_size:PadSizeType][padding bytes...][weight bytes...]
// Craft a record with size==UINT64_MAX and padding_size==0 so that
// weight_size = size-header-padding wraps and seekg performs a backward/
// out-of-bounds seek (the Finding-1 mechanism).
std::string make_malformed_weightsource_blob(uint64_t source_id) {
    std::string buf;
    // TODO: serialize tag, len=UINT64_MAX, device_id, source_id, padding_size=0.
    (void)source_id;
    return buf;
}

}  // namespace

TEST(SingleFileStorage_BuildContentIndex, MalformedWeightSourceDoesNotPoisonCacheSources) {
    const uint64_t attacker_source_id = 0xA11CEull;
    const std::string path = "crafted_weightsource.cache";

    {
        std::ofstream out(path, std::ios::binary);
        out << make_malformed_weightsource_blob(attacker_source_id);
    }

    std::ifstream in(path, std::ios::binary);
    ASSERT_TRUE(in.good());

    // TODO: construct SingleFileStorage with its shared context.
    // auto ctx = std::make_shared<SharedContext>();
    // SingleFileStorage storage(ctx);
    // const bool ok = storage.build_content_index(in);

    // Pre-fix this fails: build_content_index returns false (bad seek) yet
    // ctx->m_cache_sources already contains attacker_source_id, OR returns true
    // with a rewound stream. Post-fix the malformed record must be rejected and
    // leave the map clean.
    // EXPECT_FALSE(ok);
    // EXPECT_EQ(ctx->m_cache_sources.count(attacker_source_id), 0u);
    GTEST_SKIP() << "TODO: wire up SingleFileStorage symbols and crafted TLV blob";
}
```
**Build / run:** Build target: the inference component's gtest target (e.g. ov_inference_unit_tests — confirm from openvino/src/inference/tests). Run: ov_inference_unit_tests --gtest_filter=SingleFileStorage_BuildContentIndex.* . Pre-fix expectation: the empty m_cache_sources[source_id] entry persists after a failed parse (assertion on map.count fails) and/or build_content_index returns true on the wrapped backward seek; with ASan a follow-on blob load using the poisoned/rewound state may report heap-buffer-overflow / read of invalid offset. Post-fix: malformed record rejected, map clean, test passes.

## Suggested fix
Move the `m_shared_context->m_cache_sources[source_id] = {}` assignment to AFTER the successful seek check, matching the `blob_reader` pattern: first perform `s.seekg(...)`, then check `s.good()`, and only if true update the map. Additionally, validate `source_id` is within a reasonable range if one is defined, to prevent unbounded map growth from malformed records.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #238.
