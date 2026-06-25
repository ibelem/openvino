// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for the integer-underflow validation bypass at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:157-159.
// Pre-fix: a crafted DataHeader with consts_offset < custom_data_offset makes
//   (consts_offset - custom_data_offset) underflow, custom_data_size set to the
//   wrapped value satisfies check #2, is_valid_model becomes true, and the path
//   proceeds to construct over-sized buffers / OOB read.
// Post-fix (>= ordering guards): is_valid_model is false and process_model throws
//   ov::Exception via OPENVINO_ASSERT at :160 before any unsafe buffer use.
//
// Harness: ov_cpu_unit_tests (the intel_cpu component's gtest target).
// NOTE: ModelDeserializer / pass::StreamSerialize::DataHeader and the AlignedBuffer
//   wrapping constructor symbols must be confirmed against the real headers — TODOs below.

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

// TODO: include the real headers, e.g.:
//   #include "utils/graph_serializer/deserializer.hpp"
//   #include "openvino/pass/serialize.hpp"   // for pass::StreamSerialize::DataHeader
//   #include "openvino/runtime/aligned_buffer.hpp"

TEST(CpuModelDeserializer, RejectsUnderflowingDataHeader) {
    // sizeof(DataHeader) is 6 * sizeof(size_t) == 48 on LP64.
    // TODO: replace with the real pass::StreamSerialize::DataHeader type & field order.
    struct DataHeaderLike {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};
    static_assert(sizeof(DataHeaderLike) == 48, "layout assumption");

    hdr.custom_data_offset = sizeof(DataHeaderLike);          // == 48, passes check #1
    hdr.consts_offset      = 0;                               // < custom_data_offset -> underflow
    hdr.custom_data_size   = hdr.consts_offset - hdr.custom_data_offset; // wraps to SIZE_MAX-47
    hdr.model_offset       = 8;
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset;       // == 8
    hdr.model_size         = 0;

    std::vector<char> blob(64, 0);
    std::memcpy(blob.data(), &hdr, sizeof(hdr));

    // TODO: wrap `blob` in an ov::AlignedBuffer (or the project's SharedBuffer over a
    //       std::vector) so model_buffer->get_ptr()/size() return blob.data()/blob.size().
    // std::shared_ptr<ov::AlignedBuffer> buf = make_aligned_buffer(blob.data(), blob.size());
    //
    // ModelDeserializer deserializer(/* ctor args: model_buffer, extension, decrypt cbs */);
    // std::shared_ptr<ov::Model> model;
    // EXPECT_THROW(deserializer.process_model(model, buf), ov::Exception);
    GTEST_SKIP() << "TODO: finalize ModelDeserializer ctor/process_model invocation and AlignedBuffer wrapper";
}
