# Security finding #320: For element types with bitwidth >= 8 (the common 8/16/32/64-bit typ…

**Summary:** For element types with bitwidth >= 8 (the common 8/16/32/64-bit typ…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Consumers that rely on the return value of get_memory_size (or the get_byte_size() chain) to size allocations or bound copies will receive a drastically undersized byte count for large element counts, enabling heap overflows (CWE-787) or information disclosure (CWE-125) depending on how the undersized value is used.
**Affected location:** `targets/openvino/src/core/src/memory_util.cpp:53` — `get_memory_size()`
**Validated for repos:** openvino
**Trust boundary:** Element count (n) derived from attacker-controlled shapes passed through ITensor::get_byte_size or any direct caller

## Description / Root cause
For element types with bitwidth >= 8 (the common 8/16/32/64-bit types), get_memory_size at line 53 computes '(type.bitwidth() / 8) * n' with no overflow guard. If n is large (e.g., an attacker passes a single-dimension shape of SIZE_MAX/4), then for a 64-bit type (bitwidth()/8 == 8), the product 8 * n wraps modulo 2^64, returning a tiny byte count. By contrast, the safe overload get_memory_size_safe(type, n) at line 58-62 uses mul_overflow<size_t> to detect this, but that safe path is not called from ITensor::get_byte_size().

**Validator analysis:** The flaw is real: get_memory_size (memory_util.cpp:45-55) takes the >=8-bit branch at line 53 and multiplies bytes-per-element by n with no overflow check, while the sibling get_memory_size_safe (lines 57-63) explicitly uses mul_overflow<size_t> for exactly this case. ITensor::get_byte_size (itensor.cpp:40-42) routes through the UNSAFE variant, confirming the missing-guard claim. CWE-190 is the correct primary type, and the downstream CWE-787/125 impact is plausible where a consumer sizes a buffer from get_byte_size and then writes get_size() elements. The worked exploit arithmetic is correct (shape [SIZE_MAX/8+2], f64 -> 8 returned). For openvinoEp the chain is gated: ORT owns tensor allocation/validation, so an overflow-sized element count cannot actually be materialized and bound from the EP boundary -> rejected. For openvino core the unsafe function is directly reachable from the public get_byte_size API and from internal shape-driven sizing, so it is validated as a real latent integer-overflow in a size computation. The proposed fix (guard the line-53 multiply with mul_overflow + OPENVINO_ASSERT) is correct and sufficient; equivalently, route get_byte_size through get_memory_size_safe and convert std::nullopt into an OPENVINO_ASSERT/throw at the ITensor boundary, which is cleaner since the safe machinery already exists. Either approach turns silent wraparound into a thrown error at the API boundary.

## Exploit / Proof of Concept
Pass a shape [SIZE_MAX / 4] with element type f64 (bitwidth 64). get_memory_size(f64, SIZE_MAX/4): (64/8) * (SIZE_MAX/4) = 8 * (SIZE_MAX/4) = 2 * SIZE_MAX mod 2^64 = SIZE_MAX - 1 (still large in this case). More precisely: shape [SIZE_MAX/8 + 2] with f64 → 8 * (SIZE_MAX/8+2) wraps to 16, so get_byte_size() returns 16 bytes for a tensor holding SIZE_MAX/8+2 f64 elements, driving a multi-gigabyte OOB write if an allocation of 16 bytes is made and the tensor is then filled.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression for CWE-190 in ov::util::get_memory_size (memory_util.cpp:53).
// Pre-fix: the >=8-bit branch returns (type.bitwidth()/8)*n with no overflow
// guard, so n == SIZE_MAX/8 + 2 with f64 (bytes-per-element == 8) wraps modulo
// 2^64 to 8, silently undersizing the byte count.
// Post-fix (guard line 53 with mul_overflow + OPENVINO_ASSERT, or route through
// get_memory_size_safe): the overflow is detected and the call throws.
//
// The matching safe API get_memory_size_safe (memory_util.cpp:57-63) must keep
// returning std::nullopt for the same input.

#include <gtest/gtest.h>

#include <limits>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/type/element_type.hpp"

TEST(memory_util, get_memory_size_f64_overflow_is_rejected) {
    // n chosen so (8 * n) wraps to a tiny value modulo 2^64.
    const size_t n = (std::numeric_limits<size_t>::max() / 8) + 2;

    // The safe overload already detects the overflow today.
    EXPECT_EQ(ov::util::get_memory_size_safe(ov::element::f64, n), std::nullopt);

    // After the fix, the unsafe overload must NOT silently return a wrapped,
    // undersized byte count; it must throw instead of returning ~8 bytes.
    EXPECT_THROW(ov::util::get_memory_size(ov::element::f64, n), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests. Run: ov_core_unit_tests --gtest_filter=memory_util.get_memory_size_f64_overflow_is_rejected . Pre-fix the EXPECT_THROW fails because get_memory_size returns the wrapped value (~8) without throwing; with ASan the wrapped size later drives an OOB read/write in consumers. Post-fix (mul_overflow guard / route to get_memory_size_safe) the OPENVINO_ASSERT fires and the test passes.

## Suggested fix
In get_memory_size, guard the >= 8-bit multiplication with mul_overflow: replace 'return (type.bitwidth() / 8) * n;' with 'size_t result; OPENVINO_ASSERT(!mul_overflow<size_t>(type.bitwidth() / 8, n, result), "byte size overflow"); return result;'. Alternatively, deprecate get_memory_size in favor of get_memory_size_safe for all callers that operate on attacker-influenced element counts.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #320.
