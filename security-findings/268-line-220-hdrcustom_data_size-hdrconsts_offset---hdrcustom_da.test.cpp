// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-191 integer underflow in
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:219-232
// Pre-fix: a crafted DataHeader with consts_offset=0, custom_data_offset=48,
//   custom_data_size=0-48 (=0xFFFFFFFFFFFFFFD0) passes the equality check at line 220,
//   OPENVINO_ASSERT(222) does not fire, and resize(line 232) throws std::length_error
//   (or attempts a huge allocation) -> DoS / wrong exception type.
// Post-fix: monotonicity guard makes is_valid_model false, so OPENVINO_ASSERT throws ov::Exception.
//
// NOTE: ModelDeserializer / pass::StreamSerialize::DataHeader are internal CPU-plugin
// types; the exact include path and ctor signature must be confirmed against the source.
#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
#include "openvino/core/except.hpp"
// TODO: include the real headers exposing ModelDeserializer and StreamSerialize::DataHeader
//       (e.g. src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.hpp and
//        the serialize pass header). Confirm names before compiling.

TEST(CpuModelDeserializer, MalformedHeaderRejectedNoUnderflow) {
    // 6 x size_t header laid out as in StreamSerialize::DataHeader
    struct DataHeaderLayout {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};
    static_assert(sizeof(DataHeaderLayout) == 48, "adjust to real DataHeader layout");

    hdr.custom_data_offset = sizeof(DataHeaderLayout);          // 48
    hdr.consts_offset      = 0;                                 // underflow trigger
    hdr.custom_data_size   = (size_t)0 - sizeof(DataHeaderLayout); // 0xFFFFFFFFFFFFFFD0
    hdr.model_offset       = 49;
    hdr.consts_size        = 49;                                // == model_offset - consts_offset
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.resize(64, '\0'); // file_size > model_offset
    std::stringstream ss(blob, std::ios::in | std::ios::out | std::ios::binary);

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with the real ctor (decrypt config / model_buffer args)
    //       and invoke the istream overload of process_model(model, std::ref(ss)).
    // Expected POST-FIX behaviour: the monotonicity guard rejects the header.
    // EXPECT_THROW({ ModelDeserializer d(/*...*/); d.process_model(model, std::ref(ss)); }, ov::Exception);
    GTEST_SKIP() << "Fill in ModelDeserializer construction/invocation per source; assert ov::Exception is thrown (not std::length_error/bad_alloc).";
}
