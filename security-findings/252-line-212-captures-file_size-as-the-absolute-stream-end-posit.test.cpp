// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for absolute-vs-relative base mismatch at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:212,221,224.
// Pre-fix: when the istream is positioned at hdr_pos>0, file_size (absolute) is mixed
// with relative header offsets, so hdr.model_size is inflated by hdr_pos and the final
// read (line 250) runs past EOF leaving a zero-padded buffer / failing pugixml parse.
// Post-fix: the deserializer must reject the blob (OPENVINO_ASSERT) or size model_size
// using avail_size = file_size - hdr_pos so no over-read occurs.
//
// TODO: confirm the exact target/binary (expected: ov_cpu_unit_tests) and that
//       ModelDeserializer / pass::StreamSerialize::DataHeader are reachable from the
//       unit-test include path. ModelDeserializer is an internal CPU-plugin symbol;
//       if it is not exported to tests, exercise the bug through ov::Core::import_model
//       on a CPU-plugin blob whose stream is pre-advanced by P>0 bytes instead.
//
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "openvino/core/except.hpp"
// TODO: include the real headers for ModelDeserializer and StreamSerialize::DataHeader
//       (e.g. utils/graph_serializer/deserializer.hpp and the serialize pass header).

TEST(intel_cpu_ModelDeserializer, RejectsHeaderOffsetsPastEofWhenStreamPreAdvanced) {
    using DataHeader = ov::pass::StreamSerialize::DataHeader;  // TODO: verify namespace

    // Build a blob: [P prefix bytes][DataHeader][consts][model], but truncate so that
    // model_offset+hdr_pos lies past EOF while file_size > model_offset still holds.
    const size_t P = 64;                 // hdr_pos prefix (simulates cache metadata)
    const size_t S = 16;                 // consts_size
    DataHeader hdr{};
    hdr.custom_data_offset = sizeof(DataHeader);
    hdr.custom_data_size   = 0;
    hdr.consts_offset      = sizeof(DataHeader);
    hdr.consts_size        = S;
    hdr.model_offset       = sizeof(DataHeader) + S;
    // hdr.model_size is recomputed by the deserializer.

    std::string buf;
    buf.append(P, '\xAB');                                   // prefix => hdr_pos == P
    buf.append(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    buf.append(S, '\x00');                                  // consts region
    // Intentionally provide FEWER model bytes than model_size-relative would require,
    // so that model_offset + hdr_pos is past the real end of the (relative) data.
    buf.append(4, '<');                                     // tiny, truncated 'model'

    std::istringstream ss(buf, std::ios::binary);
    ss.seekg(static_cast<std::streamoff>(P));               // advance to hdr_pos = P

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with no-op cache_decrypt / create_ov_model hook.
    // ModelDeserializer des(...);
    // Pre-fix this either over-reads (short read, zero-padded XML) and asserts inside
    // pugixml, or silently builds a corrupt model. Post-fix it must reject cleanly.
    // EXPECT_THROW(des.process_model(model, std::ref(ss)), ov::Exception);
    GTEST_SKIP() << "TODO: wire ModelDeserializer ctor + create_ov_model callback; "
                    "assert EXPECT_THROW(process_model(model, std::ref(ss)), ov::Exception).";
}
