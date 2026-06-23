# Security finding #81: seek_current_ptr(std::streamsize size) at line 146-152 performs nei…

**Summary:** seek_current_ptr(std::streamsize size) at line 146-152 performs nei…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Two exploitation paths: (1) Oversized positive data_size (still within std::streamsize range) — seekg silently positions stream past EOF; the next read at program.cpp:2197-2198 (padding bytes via ib >> make_data(pad.data(), pad.size())) returns 0 bytes and triggers OPENVINO_ASSERT(read_size == size), crashing the inference process (DoS). (2) data_size > LLONG_MAX — _offset wraps to a huge value; get_bytes_to_sub_buffer_boundary() and get_bytes_to_page_boundary() yield wrong padding lengths; for subsequent data nodes in zero-copy mode, create_subbuffer() receives a corrupted offset, mapping an arbitrary byte range of the model tensor into GPU memory (information disclosure / type confusion in kernel parameter dispatch).
**Affected location:** `targets/openvino/src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:146` — `BinaryInputBuffer::seek_current_ptr()`
**Validated for repos:** openvino
**Trust boundary:** GPU model cache blob loaded from filesystem or OpenVINO model-caching API into cldnn::BinaryInputBuffer

## Description / Root cause
seek_current_ptr(std::streamsize size) at line 146-152 performs neither (a) an upper-bound check that current_pos+size <= stream_end nor (b) a non-negative check on size. The caller at data.hpp:439 passes data_size (a size_t read verbatim from the untrusted blob at data.hpp:416) with no validation. At line 151, _offset += size performs size_t += std::streamsize: if data_size > LLONG_MAX the implicit size_t→std::streamsize conversion yields a large negative signed value, which when added back to size_t _offset via unsigned arithmetic wraps _offset to a garbage value. Even a non-overflowing but oversized data_size silently advances seekg past EOF with no immediate error (eofbit is only set on the next sgetn call), corrupting subsequent reads.

**Validator analysis:** The flaw is real and as described. binary_buffer.hpp:146-152 seeks the istream by an attacker-influenced std::streamsize without any non-negative or upper-bound validation, then mixes signedness in `_offset += size` (_offset is size_t, line 183). data.hpp:416 reads `data_size` (size_t) verbatim from the cache blob and, in the zero-copy branch (data.hpp:437-439), passes it straight to seek_current_ptr with no validation; create_subbuffer at 438 already used ib.get_offset(), and the corrupted _offset then poisons get_bytes_to_sub_buffer_boundary()/get_bytes_to_page_boundary() (lines 157-167) for subsequent nodes. CWE-20 (Improper Input Validation) is accurate; an integer-truncation/wrap (CWE-190/197) facet also applies for data_size>LLONG_MAX. Impact is plausibly DoS via OPENVINO_ASSERT on the next failed read (read() at line 108-109), and potentially corrupted subbuffer offsets for following data nodes; full arbitrary-range mapping is harder to guarantee but the integrity corruption is real. Note the bug is only reachable on the host-accessible + zero-copy path (xe2 integrated GPU, usm_host, 4K-aligned mmap, non-weightless) per data.hpp:421-439; the non-zero-copy paths instead use `ib >> make_data(..., data_size)` which is itself bounded by the read() assert. The proposed fix is correct and sufficient: guarding size>=0 and current_pos+size<=end_pos in seek_current_ptr closes both the negative-conversion and past-EOF cases, and the call-site data_size<=streamsize::max() assert is a sound defense-in-depth. One refinement: cast size to size_t only after the >=0 check (as the fix does) to avoid the signed/unsigned mix at line 151, and consider validating data_size against the remaining stream length at data.hpp:416 right after the read so all consumers (not just the zero-copy branch) benefit.

## Exploit / Proof of Concept
Craft a cache blob where the data_size field (8 bytes at the position of data.hpp:416) is set to 0xFFFFFFFFFFFFFFFF (SIZE_MAX). When load_weights() reads it into data_size (size_t) and calls ib.seek_current_ptr(data_size): (a) std::streamsize size = -1 (after truncation/conversion), causing _stream.seekg to seek to current_pos - 1 (backward), and (b) _offset += size_t(std::streamsize(-1)) = _offset + SIZE_MAX, wrapping _offset to a garbage value. Subsequent calls to get_bytes_to_sub_buffer_boundary() return wrong padding; the next ib >> make_data(pad.data(), pad.size()) at program.cpp:2197 reads from the wrong stream position, triggering the OPENVINO_ASSERT or silently reading garbage bytes. Alternatively, set data_size to (stream_total_size - current_pos + 1) to force a silent past-EOF seek and crash on the next read.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
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
```
**Build / run:** Build target: ov_gpu_unit_tests. Run: ov_gpu_unit_tests --gtest_filter='binary_input_buffer_seek.*'. Pre-fix the two ASSERT_ANY_THROW expectations FAIL (no exception is raised; seek_current_ptr silently advances/wraps), and under ASan the negative-size case may also surface as an out-of-range seekg / corrupted _offset. Post-fix (seek_current_ptr validates size>=0 and current_pos+size<=end) both tests pass because the oversized/negative seek throws ov::Exception.

## Suggested fix
In seek_current_ptr, add two guards before the seekg: (1) assert/check that size >= 0 (reject negative values from any conversion); (2) check that current_pos + size does not exceed stream_end. Example: 'void seek_current_ptr(std::streamsize size) { OPENVINO_ASSERT(size >= 0, "[GPU] seek_current_ptr: negative size"); std::streampos current_pos = _stream.tellg(); _stream.seekg(0, std::ios::end); std::streampos end_pos = _stream.tellg(); _stream.seekg(current_pos); OPENVINO_ASSERT(static_cast<std::streamoff>(end_pos - current_pos) >= size, "[GPU] seek_current_ptr: size exceeds stream bounds"); _stream.seekg(current_pos + size); _offset += static_cast<size_t>(size); }'. At the call site in data.hpp:439, also validate data_size before passing: 'OPENVINO_ASSERT(data_size <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()), "[GPU] data_size too large for seek");'.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #81.
