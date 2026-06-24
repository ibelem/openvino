// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the missing size guard at
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:152-160
// Pre-fix: ModelDeserializer::process_model(AlignedBuffer) memcpy's sizeof(DataHeader)
//   (48 bytes) from an AlignedBuffer backing fewer bytes -> ASan heap-buffer-overflow.
// Post-fix (OPENVINO_ASSERT(file_size >= sizeof(hdr))): the call cleanly throws ov::Exception
//   before any read, so ASSERT_ANY_THROW passes and ASan stays quiet.
//
// TODO: confirm the intel_cpu unit-test target name (likely ov_cpu_unit_tests) and the
//   include path for ModelDeserializer (deserializer.hpp) from the surrounding tests/ dir.
// TODO: ModelDeserializer's ctor takes a model_buffer, weights cb, decrypt cb, origin weights;
//   verify the exact signature before relying on it.

#include <gtest/gtest.h>
#include <memory>
#include "openvino/runtime/aligned_buffer.hpp"
// TODO: #include the real deserializer header used by intel_cpu graph_serializer

TEST(CpuModelDeserializer, RejectsUndersizedCacheBlobNoOverRead) {
    // A buffer far smaller than sizeof(StreamSerialize::DataHeader) (== 48 bytes on 64-bit).
    constexpr size_t kTiny = 1;
    auto buf = std::make_shared<ov::AlignedBuffer>(kTiny, /*alignment=*/64);
    std::memset(buf->get_ptr(), 0, kTiny);

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with the real ctor and no-op decrypt callbacks,
    //   then invoke the AlignedBuffer process_model overload (directly or via operator>>).
    // Pre-fix this over-reads up to 47 bytes past buf (ASan abort);
    // post-fix it throws on the size guard.
    // ASSERT_ANY_THROW(deserializer.process_model(model, buf));
    GTEST_SKIP() << "TODO: wire up ModelDeserializer ctor + invocation (see header TODOs).";
}
