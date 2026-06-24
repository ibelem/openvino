# Security finding #199: At line 221, a raw `size_t bytes` is deserialized from an untrusted…

**Summary:** At line 221, a raw `size_t bytes` is deserialized from an untrusted…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** An attacker who can supply a crafted model-cache blob can cause the process to attempt an arbitrarily large heap allocation (up to SIZE_MAX bytes) before any decryption or integrity check is ever performed. For values in the gigabyte range this exhausts available memory (OOM / SIGKILL on Linux, out-of-memory crash on Windows). For SIZE_MAX or values exceeding `std::string::max_size()`, a `std::length_error` is thrown from the standard library, which — if uncaught in the caller — terminates or corrupts the inference engine process. Any application that loads model caches from externally supplied files or network streams is affected.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:225` — `EncryptedBinaryInputBuffer::EncryptedBinaryInputBuffer()`
**Validated for repos:** openvino
**Trust boundary:** Serialized model-cache blob read from std::istream (untrusted source) → raw size_t field decoded at line 221 → std::string allocation at line 225

## Description / Root cause
At line 221, a raw `size_t bytes` is deserialized from an untrusted stream with no value validation. At line 225, `std::string str(bytes, 0)` immediately uses that value as an allocation size with zero intervening guards — no check against remaining stream bytes, no maximum-size cap, and no ciphertext integrity verification. The `OPENVINO_ASSERT` in `BinaryInputBuffer::read` (line 109) only verifies that exactly `sizeof(size_t)` bytes were consumed from the stream; it says nothing about the decoded integer value. The `get_stream_size()` helper exists on the class (lines 132-140) but is never invoked here.

**Validator analysis:** The defect is real in openvino: at binary_buffer.hpp:221 a size_t length is deserialized from the model-cache stream with no validation, and at line 225 it is used directly as a std::string allocation size. The OPENVINO_ASSERT at line 109 only verifies sizeof(size_t) bytes were read, never the decoded value; get_stream_size() (132-140) exists but is unused here. The subsequent read at line 226 (and its assert) only fires AFTER the allocation at 225, so it does not mitigate the excessive allocation. CWE-789 (Memory Allocation with Excessive Size Value) is accurate: gigabyte values cause OOM/bad_alloc, SIZE_MAX throws std::length_error — DoS, not memory corruption (the std::string itself is bounds-safe). Impact wording ('corrupt the inference engine process') overstates it; it is a DoS/uncaught-exception abort, no corruption. The proposed fix is directionally correct: cap `bytes` and validate against remaining stream bytes before allocating. The remaining-bytes computation should use get_stream_size()/offset arithmetic already on the class rather than re-seeking, and the cap should be a documented constant. The const_cast at line 226 is a legitimate cleanup (use str.data() in C++17) but is a separate non-security nit, not part of this CWE-789 flaw. Reachability requires encrypted GPU model caching to be enabled by the host application; for plain (unencrypted) caches the non-encrypted BinaryInputBuffer path is taken, so this is conditional on the encryption callback being set.

## Exploit / Proof of Concept
Craft a model-cache blob whose first 8 bytes (the `bytes` length field) encode a large value such as 0x0000000200000000 (8 GB) or 0xFFFFFFFFFFFFFFFF (SIZE_MAX). Feed this blob to any code path that constructs `EncryptedBinaryInputBuffer`. At line 225 `std::string str(bytes, 0)` attempts to allocate that many bytes before `decrypt` is called. No prior check prevents the allocation from being attempted, resulting in process-level DoS.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build: cmake --build . --target ov_gpu_unit_tests. Run: ./ov_gpu_unit_tests --gtest_filter=encrypted_binary_input_buffer.rejects_oversized_length_field. Pre-fix expectation: under ASan/normal run the std::string(SIZE_MAX,0) at binary_buffer.hpp:225 throws std::length_error (or bad_alloc/OOM for multi-GB inputs) BEFORE any size guard; post-fix the added OPENVINO_ASSERT throws ov::Exception. Resolve the two TODOs (engine accessor + exact symbol/namespace) before compiling.

## Suggested fix
Before line 225, validate `bytes` against both a hardcoded sanity ceiling and the remaining bytes actually available in the stream. For example:
```cpp
constexpr size_t MAX_ENCRYPTED_BLOB_BYTES = 2ULL * 1024 * 1024 * 1024; // 2 GB
OPENVINO_ASSERT(bytes <= MAX_ENCRYPTED_BLOB_BYTES,
    "[GPU] Encrypted blob size exceeds maximum: " + std::to_string(bytes));
// Also guard against reading past end-of-stream:
std::streampos cur = _stream.tellg();
_stream.seekg(0, std::ios::end);
size_t remaining = static_cast<size_t>(_stream.tellg()) - static_cast<size_t>(cur);
_stream.seekg(cur);
OPENVINO_ASSERT(bytes <= remaining,
    "[GPU] Encrypted blob size exceeds remaining stream bytes");
std::string str(bytes, 0);
```
Additionally, replace the `const_cast` write at line 226 with the C++17 non-const `str.data()` to eliminate undefined behavior when writing into the string buffer.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #199.
