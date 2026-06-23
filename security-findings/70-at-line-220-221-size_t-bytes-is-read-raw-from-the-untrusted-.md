# Security finding #70: At line 220-221, `size_t bytes` is read raw from the untrusted blob…

**Summary:** At line 220-221, `size_t bytes` is read raw from the untrusted blob…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Two distinct failure modes: (1) If `bytes` is set to SIZE_MAX or any value exceeding available heap, `std::string(bytes, 0)` throws `std::bad_alloc` which is uncaught in the constructor and propagates up, crashing the inference process (DoS). (2) If `bytes` is large but allocatable yet exceeds remaining stream data, `BinaryInputBuffer::read` at line 226 calls `sgetn` which returns fewer bytes than requested, causing `OPENVINO_ASSERT` at line 109 to abort the process (DoS). In both cases, any caller loading a crafted kernel cache blob (from disk, network, or shared storage) is affected.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:220` — `EncryptedBinaryInputBuffer::EncryptedBinaryInputBuffer()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted GPU kernel cache blob stream → EncryptedBinaryInputBuffer constructor via std::istream&

## Description / Root cause
At line 220-221, `size_t bytes` is read raw from the untrusted blob stream (8 bytes on 64-bit) with no upper-bound or sanity check. At line 225, `std::string str(bytes, 0)` uses this attacker-controlled value directly as the allocation size. No comparison against remaining stream bytes, `get_stream_size()`, `_offset`, or any other guard is performed between lines 221 and 225. `get_stream_size()` exists (lines 132-139) but is never called on this path.

**Validator analysis:** The flaw is real and as described: at binary_buffer.hpp:220-221 `bytes` is read raw from the (potentially tampered) encrypted GPU cache blob, and at line 225 `std::string str(bytes, 0)` allocates that many bytes with zero validation; get_stream_size()/get_offset() (lines 132-144) exist but are never consulted here. CWE-789 (Memory Allocation with Excessive Size Value) is the accurate categorisation. The impact is largely accurate but the second mode's phrasing is slightly off: OPENVINO_ASSERT at line 109 THROWS ov::Exception (not std::abort); both modes are therefore an uncaught-exception DoS if the caller doesn't wrap the load — not memory corruption. Reachability is real but gated: this path only runs when cache encryption is enabled (a non-null decrypt callback, line 215-218), which is opt-in within OpenVINO and not driven by the ORT OpenVINO EP, hence the EP rejection. The proposed fix is correct and sufficient: bounding `bytes` against `get_stream_size() - get_offset()` via OPENVINO_ASSERT before allocation closes both the bad_alloc and the short-read cases; an additional hard MAX_KERNEL_CACHE_SIZE cap is a sensible defence-in-depth. The secondary suggestion to replace `const_cast<void*>(reinterpret_cast<const void*>(str.c_str()))` with `str.data()` (C++17 non-const overload) is a valid correctness improvement (avoids writing through a const-qualified pointer), though it is not the security-critical change.

## Exploit / Proof of Concept
Craft a blob where the 8-byte little-endian `size_t` field encoding the ciphertext length is set to 0xFFFFFFFFFFFFFFFF (SIZE_MAX). When `EncryptedBinaryInputBuffer` is constructed with this stream, line 225 executes `std::string str(0xFFFFFFFFFFFFFFFF, 0)`, which attempts an ~18 EB heap allocation, throws `std::bad_alloc`, and crashes the process. Alternatively, set `bytes` to e.g. 0x10000000 (256 MB) while providing only 8 bytes of ciphertext data after the length field; the string allocation succeeds, but `BinaryInputBuffer::read` at line 226 reads only 8 bytes and `OPENVINO_ASSERT(read_size == size)` at line 109 fires, aborting the process.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
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
```
**Build / run:** Build target: ov_gpu_unit_tests. Run: ov_gpu_unit_tests --gtest_filter='encrypted_binary_input_buffer.*'. Pre-fix expectation: 'rejects_excessive_size_field' aborts/crashes via uncaught std::bad_alloc (or under ASan an allocation-size-too-big report), and 'rejects_truncated_payload' currently throws ov::Exception from the line-109 OPENVINO_ASSERT; after the proposed bounds check both tests pass by throwing ov::Exception cleanly. Verify get_test_engine() helper name against the surrounding intel_gpu/tests/unit test_utils.h before use.

## Suggested fix
After reading `bytes` at line 221, validate it against the remaining stream length before allocating. Compute remaining bytes as `get_stream_size() - get_offset()` (noting that `get_stream_size()` returns total size from start, so remaining = `get_stream_size() - _offset`), then add: `const size_t remaining = get_stream_size() - get_offset(); OPENVINO_ASSERT(bytes <= remaining, "[GPU] Encrypted blob size field exceeds stream length");` Additionally, consider a hard upper-bound sanity check (e.g., `bytes <= MAX_KERNEL_CACHE_SIZE`) to guard against resource exhaustion even when the stream is long enough. Also replace `const_cast<void*>(reinterpret_cast<const void*>(str.c_str()))` on line 226 with `static_cast<void*>(str.data())` (C++17 non-const `data()`) to avoid undefined behavior when writing through a const-cast pointer.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #70.
