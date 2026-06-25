// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-125 OOB read at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:154
// Pre-fix: ModelDeserializer::process_model(AlignedBuffer) memcpy's sizeof(DataHeader)
// (~48 bytes) out of a buffer whose size() may be < sizeof(hdr) -> heap OOB read
// (ASan: heap-buffer-overflow READ). Post-fix: an explicit size guard throws ov::Exception
// ("Cache blob too small...") before the memcpy, so EXPECT_THROW passes cleanly.
//
// SKELETON: ModelDeserializer is an internal CPU-plugin type; exact include path,
// constructor signature, and the AlignedBuffer concrete type must be confirmed against
// the source tree before this compiles.

#include <gtest/gtest.h>
#include <memory>
#include "openvino/runtime/aligned_buffer.hpp"   // TODO: confirm header for ov::AlignedBuffer
// TODO: include the real header that declares intel_cpu ModelDeserializer
// e.g. "utils/graph_serializer/deserializer.hpp" (path relative to intel_cpu/src)

using namespace ov;

TEST(cpu_model_deserializer, undersized_cache_blob_is_rejected_not_oob) {
    // 1-byte buffer, far smaller than sizeof(pass::StreamSerialize::DataHeader) (~48 bytes)
    auto tiny = std::make_shared<ov::AlignedBuffer>(/*size=*/1);  // TODO: confirm ctor
    reinterpret_cast<char*>(tiny->get_ptr())[0] = 0x00;

    std::shared_ptr<ov::Model> model;
    // TODO: confirm ModelDeserializer ctor args (decrypt cfg / origin weights may be required)
    // intel_cpu::ModelDeserializer des(/*...*/);
    // Pre-fix this memcpy reads 47 bytes OOB (ASan trap); post-fix it must throw.
    // EXPECT_THROW(des.process_model(model, tiny), ov::Exception);
    GTEST_SKIP() << "TODO: wire ModelDeserializer ctor + process_model(AlignedBuffer) overload";
}
