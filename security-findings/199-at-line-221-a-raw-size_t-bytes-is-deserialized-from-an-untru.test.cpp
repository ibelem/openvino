// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-789 at
// openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:220-225
// Encodes the fix: an attacker-controlled `bytes` length field in an encrypted
// model-cache blob must be rejected (throw ov::Exception) BEFORE a std::string of
// that size is allocated, instead of attempting an arbitrarily large heap alloc.
//
// Pre-fix: std::string str(bytes, 0) at line 225 attempts to allocate `bytes`
//          (e.g. SIZE_MAX -> std::length_error; multi-GB -> bad_alloc/OOM) before
//          any cap or stream-remaining check.
// Post-fix: the added OPENVINO_ASSERT(bytes <= cap && bytes <= remaining) throws
//           ov::Exception, which this test asserts.
//
// TODO(harness): place under openvino/src/plugins/intel_gpu/tests/unit/ and add to
//   the ov_gpu_unit_tests target. Confirm the exact include path and the helper to
//   obtain a test `engine&` (see existing serialization/graph unit tests for the
//   get_test_engine()/tests::engine() accessor used by that tree).
// TODO(symbols): verify namespace of EncryptedBinaryInputBuffer (cldnn) and the
//   ov::Exception type / ASSERT macro names against the real test sources.

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
// TODO: include the test engine helper header used by intel_gpu/tests/unit
// #include "test_utils.h"

using namespace cldnn;

TEST(encrypted_binary_input_buffer, rejects_oversized_length_field) {
    // Craft a blob whose first sizeof(size_t) bytes encode a huge length.
    const size_t malicious_bytes = static_cast<size_t>(-1); // SIZE_MAX
    std::string blob(reinterpret_cast<const char*>(&malicious_bytes), sizeof(malicious_bytes));
    // (no ciphertext follows — the allocation/validation must fail before any read)
    std::istringstream stream(blob, std::ios::binary);

    auto identity_decrypt = [](const std::string& s) { return s; };

    // TODO: replace with the unit tree's engine accessor.
    engine& eng = /* TODO: get_test_engine() */ *static_cast<engine*>(nullptr);

    // Pre-fix this either OOMs or throws std::length_error from std::string;
    // post-fix it must throw ov::Exception from the size validation guard.
    ASSERT_ANY_THROW({ EncryptedBinaryInputBuffer buf(stream, eng, identity_decrypt); });
}
