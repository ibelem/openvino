// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for openvino intel_cpu deserializer.cpp:219-243 (process_model istream overload).
// Pre-fix: a crafted StreamSerialize::DataHeader with consts_offset=sizeof(hdr), model_offset=0
// underflows (model_offset - consts_offset) to ~SIZE_MAX, passes is_valid_model (line 222),
// then ov::Tensor(u8, Shape({hdr.consts_size})) at line 240 attempts a ~SIZE_MAX allocation -> std::bad_alloc.
// Post-fix: header ordering/upper-bound validation must reject the blob with ov::Exception before allocation.
//
// Target: ov_cpu_unit_tests (component test tree under src/plugins/intel_cpu/tests/unit/).
// NOTE: ModelDeserializer::process_model is private; the realistic entry point is
// ov::Core::import_model(stream, "CPU") with a cache blob, OR exercising ModelDeserializer directly.
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
#include "openvino/runtime/core.hpp"

namespace {
// TODO: replace with the real layout of ov::pass::StreamSerialize::DataHeader
// (read src/core/.../pass/serialize/stream_serialize.hpp to confirm exact field order/types).
struct DataHeaderMirror {
    uint64_t custom_data_offset;
    uint64_t custom_data_size;
    uint64_t consts_offset;
    uint64_t consts_size;
    uint64_t model_offset;
    uint64_t model_size;
};

TEST(CpuModelDeserializer, UnderflowedConstsSizeIsRejected) {
    DataHeaderMirror hdr{};
    hdr.custom_data_offset = sizeof(DataHeaderMirror);   // must equal sizeof(hdr)
    hdr.custom_data_size   = 0;                          // == consts_offset - custom_data_offset
    hdr.consts_offset      = sizeof(DataHeaderMirror);
    hdr.model_offset       = 0;                          // forces underflow below
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset; // wraps to ~SIZE_MAX, == model_offset-consts_offset
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.append(64, '\0'); // ensure file_size > model_offset(=0)
    std::stringstream ss(blob);

    // TODO: confirm import_model signature/device for the cache-deserialize path that lands in
    // ModelDeserializer::process_model(istream). Pre-fix this throws std::bad_alloc at deserializer.cpp:240;
    // post-fix it must throw ov::Exception from the validated header guard instead of allocating.
    ov::Core core;
    EXPECT_THROW(core.import_model(ss, "CPU"), ov::Exception);
}
} // namespace
