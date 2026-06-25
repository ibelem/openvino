# Security finding #332: `util::aligned_alloc(m_byte_size, alignment)` at line 19 returns `n…

**Summary:** `util::aligned_alloc(m_byte_size, alignment)` at line 19 returns `n…

**CWE IDs:** CWE-476: NULL Pointer Dereference
**Severity / Impact:** Guaranteed crash (SIGSEGV / access violation) whenever `aligned_alloc` cannot satisfy a large allocation request, making OOM conditions exploitable as a reliable remote DoS against any process that reads attacker-supplied IR models.
**Affected location:** `targets/openvino/src/core/src/runtime/aligned_buffer.cpp:18` — `AlignedBuffer::AlignedBuffer(size_t, size_t)()`
**Validated for repos:** openvino
**Trust boundary:** Same trust boundary: attacker-controlled XML shape → uncapped `byte_size` → `AlignedBuffer` constructor

## Description / Root cause
`util::aligned_alloc(m_byte_size, alignment)` at line 19 returns `nullptr` on allocation failure (POSIX `posix_memalign`, Win32 `_aligned_malloc`, and most libc `aligned_alloc` variants return null rather than throwing). The constructor stores the result directly into `m_aligned_buffer` with no null check. Back in `constant.cpp:253`, `std::memset(m_data->get_ptr(), 0, m_data->size())` is called unconditionally when `memset_allocation` is true (the deserialization path sets this for string constants) or `*byte_size == 0`, passing the null pointer as the destination — immediate crash.

**Validator analysis:** Confirmed by reading: ov::util::aligned_alloc is declared noexcept and documented to return nullptr on failure (memory.hpp:62-70), and AlignedBuffer's ctor (aligned_buffer.cpp:18-20) assigns that result to m_aligned_buffer with no null check. Constant::allocate_buffer (constant.cpp:239-257) computes byte_size via get_memory_size_safe (only overflow-guarded by OPENVINO_ASSERT at 243, not capped to available memory), constructs AlignedBuffer, then on the memset_allocation/byte_size==0 branch calls std::memset(m_data->get_ptr(),0,size) — a guaranteed nullptr write when the allocation failed. An attacker can craft an IR shape whose product is a huge-but-non-overflowing size_t that aligned_alloc reliably cannot satisfy, making the null-deref a reliable crash rather than a catchable bad_alloc. CWE-476 is accurate; the 'reliable remote DoS via crafted IR' impact is fair (the size can be chosen to exceed addressable memory, so it does not depend on incidental memory pressure). Even outside the memset branch, the null pointer is stored in m_data and dereferenced on later tensor access. The proposed fix (if (!m_aligned_buffer) throw std::bad_alloc{}; after line 19) is correct and sufficient to convert the silent null into a catchable exception, aborting Constant construction before any deref; pairing it with an allocation-size cap is sound defense-in-depth. Note alignment==0 is handled (alignof(max_align_t)) and byte_size==0 yields a 1-byte alloc, so those edge cases do not change the verdict. openvinoEp is rejected because the defect and its trust boundary are fully contained in OpenVINO core and no EP plugin_impl source is on the cited path.

## Exploit / Proof of Concept
Same crafted XML as above triggers the uncapped 8 GB allocation. On a system with < 8 GB of free contiguous aligned memory, `aligned_alloc` returns null. `m_aligned_buffer` is null. Control returns to `allocate_buffer`; if the branch at line 252 (`memset_allocation || *byte_size == 0`) is taken (it is taken for string constants), `std::memset(nullptr, 0, size)` causes an immediate crash. Even for non-string numerics the null pointer is stored in `m_data` and will be dereferenced on any subsequent tensor access.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-476 null-pointer deref in
//   openvino/src/core/src/runtime/aligned_buffer.cpp:18-20
// Pre-fix: util::aligned_alloc returns nullptr on a request it cannot satisfy
//   (documented noexcept nullptr-on-failure, common/util .../memory.hpp:62-70),
//   the ctor stores it unchecked, and ov::op::Constant::allocate_buffer
//   (constant.cpp:253) memsets the null pointer -> SIGSEGV/AV under ASan.
// Post-fix: the ctor throws std::bad_alloc, so construction fails cleanly and
//   the assertions below pass.
//
// TODO(build): confirm the exact gtest target for core (likely ov_core_unit_tests)
//   and the correct include path for openvino/runtime/aligned_buffer.hpp.

#include <gtest/gtest.h>

#include <limits>
#include <new>

#include "openvino/runtime/aligned_buffer.hpp"

// Requesting a size that cannot be backed by physical/virtual memory but does
// not overflow size_t. Pre-fix this yields a stored nullptr; the fix must turn
// the failed allocation into a std::bad_alloc instead of a null buffer.
TEST(aligned_buffer, alloc_failure_throws_instead_of_null) {
    constexpr size_t huge = std::numeric_limits<size_t>::max() / 2;
    EXPECT_THROW({ ov::AlignedBuffer buf(huge, 64); }, std::bad_alloc);
}
```
**Build / run:** Build target: ov_core_unit_tests (OpenVINO core unit tests). Run: ov_core_unit_tests --gtest_filter='aligned_buffer.alloc_failure_throws_instead_of_null'. Pre-fix expectation: with ASan/standard run the failed allocation leaves m_aligned_buffer == nullptr and (when reached via Constant::allocate_buffer memset path) ASan reports a SEGV / 'WRITE of size N ... unknown address 0x000000000000' on std::memset; the EXPECT_THROW also fails because no exception is thrown. Post-fix expectation: ctor throws std::bad_alloc and the test passes.

## Suggested fix
After `util::aligned_alloc` at line 19, check for null and throw `std::bad_alloc`: `if (!m_aligned_buffer) throw std::bad_alloc{};`. This converts the silent null into a catchable C++ exception, letting callers handle OOM gracefully instead of crashing. Combined with the allocation-size cap proposed above, this provides defense-in-depth.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #332.
