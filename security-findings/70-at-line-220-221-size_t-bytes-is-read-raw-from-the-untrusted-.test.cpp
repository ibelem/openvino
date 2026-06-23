// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in
//   openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:220-226
// EncryptedBinaryInputBuffer reads an 8-byte attacker-controlled `size_t bytes`
// from the stream and immediately does `std::string str(bytes, 0)` with no
// bound vs the remaining stream length. A crafted blob with bytes==SIZE_MAX
// triggers std::bad_alloc; a large-but-truncated blob trips OPENVINO_ASSERT at
// line 109. This test encodes that a hostile size field must be rejected by an
// exception rather than crashing the process.
//
// PRE-FIX:  std::string(SIZE_MAX,0) -> std::bad_alloc escapes ctor (uncaught
//           crash under the real loader); the truncated case -> ov::Exception.
// POST-FIX: an OPENVINO_ASSERT(bytes <= get_stream_size()-get_offset()) makes
//           both crafted blobs throw ov::Exception deterministically.
//
// Harness: ov_gpu_unit_tests (gtest). Place under
//   src/plugins/intel_gpu/tests/unit/module_tests/ (e.g. binary_buffer_test.cpp).

#include "test_utils.h"                 // tests::get_test_engine()
#include "intel_gpu/graph/serialization/binary_buffer.hpp"

#include <sstream>
#include <string>
#include <limits>
#include <cstring>

using namespace cldnn;
using namespace ::tests;

namespace {

// Build a fake encrypted-cache stream: 8-byte little-endian size field followed
// by `payload` bytes of "ciphertext".
std::string make_encrypted_blob(size_t declared_size, size_t payload_len) {
    std::string blob;
    blob.append(reinterpret_cast<const char*>(&declared_size), sizeof(declared_size));
    blob.append(payload_len, '\0');
    return blob;
}

// identity decrypt callback (non-null so the ctor's OPENVINO_ASSERT(decrypt) passes)
std::string identity_decrypt(const std::string& s) { return s; }

} // namespace

// Mode 1: SIZE_MAX size field -> excessive allocation must be rejected, not crash.
TEST(encrypted_binary_input_buffer, rejects_excessive_size_field) {
    auto& engine = get_test_engine();
    const std::string blob = make_encrypted_blob(std::numeric_limits<size_t>::max(),
                                                 /*payload_len=*/8);
    std::stringstream ss(blob);
    ASSERT_ANY_THROW({
        EncryptedBinaryInputBuffer buf(ss, engine, identity_decrypt);
    });
}

// Mode 2: large declared size but truncated payload -> must throw, not abort.
TEST(encrypted_binary_input_buffer, rejects_truncated_payload) {
    auto& engine = get_test_engine();
    const std::string blob = make_encrypted_blob(/*declared_size=*/0x10000000ull,
                                                 /*payload_len=*/8);
    std::stringstream ss(blob);
    ASSERT_ANY_THROW({
        EncryptedBinaryInputBuffer buf(ss, engine, identity_decrypt);
    });
}
