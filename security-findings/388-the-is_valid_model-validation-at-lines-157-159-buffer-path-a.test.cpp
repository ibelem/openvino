// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for the integer-underflow in
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:158,220
// A crafted 49-byte cache blob with custom_data_offset=48, consts_offset=0,
// custom_data_size=0xFFFFFFFFFFFFFFD0 (= 0 - 48 wrap) passes is_valid_model pre-fix
// and then drives pugixml load_buffer / std::string::resize with ~18 EiB,
// causing bad_alloc/length_error (DoS). After the fix the header must be rejected
// via OPENVINO_ASSERT ("Could not deserialize by device xml header").
//
// SKELETON: ModelDeserializer is an internal intel_cpu class; exact test target,
// include path, and ctor signature must be confirmed against the cpu unit-test tree
// (ov_cpu_unit_tests). Filling the TODOs below is required before this compiles.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// TODO: confirm these headers/paths in src/plugins/intel_cpu/tests/unit/
// #include "utils/graph_serializer/deserializer.hpp"
// #include "openvino/runtime/aligned_buffer.hpp"
// #include "openvino/pass/serialize.hpp"  // ov::pass::StreamSerialize::DataHeader

namespace {

TEST(ModelDeserializerHeaderTest, RejectsUnderflowedCustomDataSize) {
    // 48-byte DataHeader (6 x uint64 little-endian) + 1 payload byte = 49 bytes
    struct RawHeader {
        uint64_t custom_data_offset;
        uint64_t custom_data_size;
        uint64_t consts_offset;
        uint64_t consts_size;
        uint64_t model_offset;
        uint64_t model_size;
    } hdr{};
    hdr.custom_data_offset = sizeof(RawHeader);            // 48 == sizeof(DataHeader)
    hdr.consts_offset      = 0;                             // < custom_data_offset -> underflow
    hdr.custom_data_size   = static_cast<uint64_t>(0) - sizeof(RawHeader); // 0xFFFFFFFFFFFFFFD0
    hdr.consts_size        = 0;                             // == model_offset - consts_offset
    hdr.model_offset       = 0;
    hdr.model_size         = 0;

    std::vector<uint8_t> blob(sizeof(RawHeader) + 1, 0);
    std::memcpy(blob.data(), &hdr, sizeof(RawHeader));

    // TODO: wrap blob in ov::AlignedBuffer and construct ModelDeserializer with a
    //       null/identity CacheDecrypt and a stub ov::ICore, then invoke the
    //       deserialize entry that calls process_model(...).
    // auto buffer = /* make AlignedBuffer over blob */;
    // ov::intel_cpu::ModelDeserializer des(buffer, core, /*decrypt*/{}, false, "");
    // std::shared_ptr<ov::Model> model;
    // Pre-fix: passes is_valid_model, then bad_alloc/length_error from the
    //          18-EiB load_buffer/resize. Post-fix: OPENVINO_ASSERT throws ov::Exception.
    // EXPECT_THROW(des >> model, ov::Exception);
    GTEST_SKIP() << "Fill TODOs: AlignedBuffer wrap + ModelDeserializer ctor/entry.";
}

}  // namespace
