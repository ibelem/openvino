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
