# Security finding #333: At line 259, `static_cast<size_t>(file_size - offset_in_bytes)` sil…

**Summary:** At line 259, `static_cast<size_t>(file_size - offset_in_bytes)` sil…

**CWE IDs:** CWE-197: Numeric Truncation Error
**Severity / Impact:** On 32-bit hosts: (1) If `file_size - offset_in_bytes` is an exact multiple of 2^32 (e.g., 8 GB file, 0 offset), `available_size` wraps to 0. With a static partial shape, `resolve_static_shape` (line 186) fires `OPENVINO_ASSERT(requested_size && *requested_size <= available_size)` — DoS/crash. With a dynamic partial shape, a zero-element tensor is silently returned — silent data corruption. (2) If the difference merely overflows (e.g., 4 GB + 256 bytes → truncates to 256), `resolve_static_shape` derives an undersized shape, allocates a tiny tensor and reads only 256 bytes, producing wrong model weights with no error — silent incorrect inference results (CWE-125 info leakage or incorrect computation).
**Affected location:** `targets/openvino/src/core/src/runtime/tensor.cpp:259` — `read_tensor_data(const std::filesystem::path&, …)()`
**Validated for repos:** openvino
**Trust boundary:** Caller-supplied `file_name` and `offset_in_bytes` → `std::filesystem::file_size()` returns `uintmax_t` (64-bit); result narrowed to `size_t` (32-bit on 32-bit hosts) without a range guard

## Description / Root cause
At line 259, `static_cast<size_t>(file_size - offset_in_bytes)` silently truncates the 64-bit difference to 32 bits on a 32-bit host. The preceding assertion at line 254 only verifies `offset_in_bytes <= file_size` (comparing `size_t` to `uintmax_t`); it does NOT check that `(file_size - offset_in_bytes) <= SIZE_MAX`, so the assertion passes even when the difference exceeds 4 GB. The truncated value is then passed as `available_size` to `resolve_static_shape` at line 260.

**Validator analysis:** The cited code is accurate: line 253 reads file_size as uintmax_t (64-bit), the only guard (254) checks offset<=file_size, and line 259 narrows the 64-bit difference to size_t. On a 32-bit host (size_t=32-bit) a difference >4GB silently truncates. CWE-197 (Numeric Truncation) is the correct categorization. However the impact narrative is overstated on two points: (1) the 'DoS/crash' case is not a crash — resolve_static_shape's OPENVINO_ASSERT at line 186 throws a controlled ov::Exception that the public API converts to an error; (2) the static-shape 4GB+N 'wrong weights' case cannot occur because a 32-bit process cannot represent/allocate the full >4GB tensor in the first place (requested_size is itself a size_t). The genuinely valid residual risk is the DYNAMIC partial-shape path: a truncated available_size yields an undersized derived dimension (lines 174-182), allocates a small tensor, and read_tensor_via_istream's gcount check (200) compares against that same small byte size, so it succeeds silently — wrong/short weights with no error. This is real but strictly 32-bit-only; on 64-bit (the dominant build) size_t==uintmax_t and the cast is a no-op, so the finding has no effect there. The proposed fix (explicit `OPENVINO_ASSERT(raw_available <= numeric_limits<size_t>::max())` before the cast) is correct, minimal, matches the surrounding error-handling style, and is a no-op on 64-bit — sufficient. A regression test is not realistically compilable: it requires a 32-bit build plus an >4GB file, neither feasible in the core gtest harness, so only a skeleton is provided.

## Exploit / Proof of Concept
On a 32-bit build of OpenVINO, pass a crafted file of exactly 8 GB (or any size that is a multiple of 4 GB) and `offset_in_bytes = 0`. `file_size` = 0x200000000 (uintmax_t). Line 254 assert: `0 <= 0x200000000` → passes. Line 259: `static_cast<size_t>(0x200000000 - 0)` = `static_cast<size_t>(0x200000000)` = `0` (uint32 wrap). `resolve_static_shape(0, element_type, static_shape)` → line 186 OPENVINO_ASSERT triggers → crash/DoS. Alternatively, use a 4 GB + N byte file (N small) and a dynamic partial shape to obtain a tensor of N/element_size elements while the caller expects the full 4 GB+N tensor — incorrect model outputs.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-197 truncation at
//   openvino/src/core/src/runtime/tensor.cpp:259
//   `static_cast<size_t>(file_size - offset_in_bytes)`
// Pre-fix: on a 32-bit host a (file_size - offset) > SIZE_MAX silently
// truncates, producing an undersized available_size and silent short reads.
// Post-fix: an explicit OPENVINO_ASSERT(raw_available <= SIZE_MAX) throws.
//
// NOTE: This defect is 32-bit-only and needs a >4GB on-disk file, so it
// cannot be exercised on a normal 64-bit CI host. This is therefore a
// SKELETON; it documents the intended assertion rather than a runnable case.

#include <gtest/gtest.h>
#include "openvino/runtime/tensor.hpp"
#include "openvino/core/except.hpp"

// TODO: confirm the exact ov_core_unit_tests target and the public/internal
//       symbol that exposes ov::read_tensor_data (it lives in an anonymous /
//       internal namespace in tensor.cpp and may not be directly linkable).
TEST(read_tensor_data, region_exceeding_size_t_is_rejected_not_truncated) {
    // TODO: requires a 32-bit build (sizeof(size_t)==4) AND a sparse file
    //       whose size minus offset exceeds SIZE_MAX (e.g. 8 GB, offset 0).
    //       On 64-bit hosts the cast is a no-op and this test is vacuous.
#if SIZE_MAX < UINTMAX_MAX
    const std::filesystem::path huge_file = /* TODO: create 8GB sparse file */ "huge.bin";
    EXPECT_THROW(
        ov::read_tensor_data(huge_file, ov::element::u8, ov::PartialShape::dynamic(), /*offset=*/0, /*mmap=*/false),
        ov::Exception);
#else
    GTEST_SKIP() << "size_t is 64-bit on this host; truncation path unreachable.";
#endif
}
```
**Build / run:** Build target: ov_core_unit_tests (32-bit/armhf toolchain required). Run: ov_core_unit_tests --gtest_filter='read_tensor_data.region_exceeding_size_t_is_rejected_not_truncated'. Pre-fix expectation on a 32-bit build with an 8GB input: no throw and a silently-undersized tensor (test FAILS / silent corruption); post-fix: OPENVINO_ASSERT 'File region exceeds addressable size_t range' throws ov::Exception (test PASSES). Skipped on 64-bit hosts.

## Suggested fix
After line 253, add an explicit range check before the cast:
```cpp
const auto raw_available = file_size - offset_in_bytes;  // uintmax_t
OPENVINO_ASSERT(raw_available <= std::numeric_limits<size_t>::max(),
    "File region exceeds addressable size_t range: ", raw_available, " bytes");
const auto available_size = static_cast<size_t>(raw_available);
```
This converts what is currently a silent truncation (or wrap-to-zero) into a clear assertion failure with a diagnostic message, consistent with the existing error-handling style. On 64-bit hosts the assert is a no-op and incurs no cost.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #333.
