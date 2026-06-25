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
