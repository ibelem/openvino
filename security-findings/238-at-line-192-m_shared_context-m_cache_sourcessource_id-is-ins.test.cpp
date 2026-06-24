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