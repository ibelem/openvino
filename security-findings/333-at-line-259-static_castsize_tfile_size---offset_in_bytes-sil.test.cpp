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