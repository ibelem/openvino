// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-20 in
//   openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:146-152
// BinaryInputBuffer::seek_current_ptr(std::streamsize) performs no non-negative
// check and no current_pos+size<=stream_end check, and does `_offset += size`
// (size_t += std::streamsize). An untrusted data_size (data.hpp:416) reaches it
// unvalidated at data.hpp:439.
//
// What this encodes:
//  * Pre-fix: seek_current_ptr(huge) silently advances past EOF / wraps _offset;
//    no exception is thrown (assertions below FAIL).
//  * Post-fix: the added OPENVINO_ASSERT(size>=0) and bounds check reject the
//    oversized seek, so the call throws ov::Exception (assertions PASS).

#include "test_utils.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"

#include <sstream>
#include <limits>
#include <string>

using namespace cldnn;
using namespace ::tests;

// Oversized positive size: seek must be rejected, not silently advanced past EOF.
TEST(binary_input_buffer_seek, rejects_size_past_eof) {
    auto& engine = get_test_engine();
    std::string payload(64, '\0');           // 64-byte backing stream
    std::stringstream ss(payload);
    BinaryInputBuffer ib(ss, engine);

    // current_pos == 0, stream has only 64 bytes; a 1<<20 seek is past EOF.
    ASSERT_ANY_THROW(ib.seek_current_ptr(static_cast<std::streamsize>(1) << 20));
}

// data_size > LLONG_MAX truncates to a negative std::streamsize and wraps _offset.
// Post-fix the negative-size guard rejects it.
TEST(binary_input_buffer_seek, rejects_negative_converted_size) {
    auto& engine = get_test_engine();
    std::string payload(64, '\0');
    std::stringstream ss(payload);
    BinaryInputBuffer ib(ss, engine);

    // Emulate the implicit size_t -> std::streamsize conversion of SIZE_MAX.
    size_t data_size = std::numeric_limits<size_t>::max();
    ASSERT_ANY_THROW(ib.seek_current_ptr(static_cast<std::streamsize>(data_size)));
    // _offset must not have been corrupted by a wrapping add.
    ASSERT_EQ(ib.get_offset(), static_cast<size_t>(0));
}