// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 in
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:240 (is_valid_model gate at 219-221)
// Pre-fix: a crafted StreamSerialize blob whose header has model_offset < consts_offset
// makes hdr.consts_size underflow (size_t) to ~2^64; line 240 attempts a multi-GB/2^64
// ov::Tensor allocation -> std::bad_alloc / OOM before any stream read.
// Post-fix: the bounds/underflow check rejects the header via ov::Exception.
//
// NOTE: ModelDeserializer + pass::StreamSerialize::DataHeader are intel_cpu-internal; this is a
// SKELETON because the exact DataHeader field layout and the deserializer entry symbol must be
// confirmed from the source before this will compile.
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
// TODO: include the real headers, e.g.
//   #include "utils/serialize.hpp"            // pass::StreamSerialize::DataHeader
//   #include "utils/graph_serializer/deserializer.hpp"  // ov::intel_cpu::ModelDeserializer

TEST(IntelCpuModelDeserializer, RejectsUnderflowedConstsSize) {
    // TODO: replace with the real DataHeader type/layout from serialize.hpp
    struct DataHeaderMock {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};

    hdr.custom_data_offset = sizeof(DataHeaderMock); // must == sizeof(hdr)
    hdr.custom_data_size   = 0;                      // => consts_offset == custom_data_offset
    hdr.consts_offset      = sizeof(DataHeaderMock);
    hdr.model_offset       = 0;                      // model_offset < consts_offset -> underflow
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset; // wraps to ~2^64
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.resize(sizeof(hdr) + 16, '\0'); // file_size > model_offset, tiny file
    std::stringstream ss(blob);

    std::shared_ptr<ov::Model> model;
    // TODO: construct the deserializer exactly as production code does and invoke the istream overload:
    //   ov::intel_cpu::ModelDeserializer d(ss, /*decrypt*/{}, /*model_loader*/{}, /*orig_weights*/nullptr);
    //   EXPECT_THROW(d >> model, ov::Exception);   // post-fix: header rejected, no huge allocation
    // Pre-fix this path reaches make_shared<ov::Tensor>(u8,{~2^64}) -> std::bad_alloc (ASan/OOM).
    GTEST_SKIP() << "TODO: wire up real ModelDeserializer entry point and DataHeader layout";
}
