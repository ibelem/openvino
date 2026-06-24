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
