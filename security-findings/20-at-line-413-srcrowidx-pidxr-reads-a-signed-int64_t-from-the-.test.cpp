// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-129 OOB read in ov::npuw::util::gather
//   src: openvino/src/plugins/intel_npu/src/plugin/npuw/util.cpp:413-415
//        auto srcRowIdx = pIdx[r];
//        auto pSrcRow = pSrc + src_shape[1] * srcRowIdx * src_type.size();   // <-- no bounds check
//        std::copy_n(pSrcRow, src_shape[1] * src_type.size(), pDst);
//
// Pre-fix: a negative or >=src_shape[0] index makes pSrcRow point outside the
//   src allocation; std::copy_n reads OOB -> ASan heap-buffer-overflow (or SIGSEGV).
// Post-fix (proposed): gather() validates 0 <= idx < src_shape[0] and throws
//   ov::Exception, so EXPECT_THROW(..., ov::Exception) passes and no OOB occurs.
//
// Harness: ov_npu_unit_tests (gtest + ASan). util.cpp is already in OBJECT_FILES
// of tests/unit/CMakeLists.txt, and util.hpp is on the include path.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace {

// src: 4 rows x 8 cols of f16; valid index range is [0, 4).
static ov::SoPtr<ov::ITensor> make_src() {
    ov::Tensor t(ov::element::f16, ov::Shape{4, 8});
    std::fill_n(static_cast<uint8_t*>(t.data()), t.get_byte_size(), uint8_t{0});
    return ov::get_tensor_impl(t);
}

// idx: shape {1, N} of i64 carrying a single attacker-controlled lookup value.
static ov::SoPtr<ov::ITensor> make_idx(int64_t value) {
    ov::Tensor t(ov::element::i64, ov::Shape{1, 1});
    t.data<int64_t>()[0] = value;
    return ov::get_tensor_impl(t);
}

// dst: shape {1, N, 8} of f16 (src_shape[1] == dst_shape[2]).
static ov::SoPtr<ov::ITensor> make_dst() {
    ov::Tensor t(ov::element::f16, ov::Shape{1, 1, 8});
    return ov::get_tensor_impl(t);
}

// A valid, in-range index must succeed.
TEST(NpuwUtilGatherBounds, InRangeIndexSucceeds) {
    EXPECT_NO_THROW(ov::npuw::util::gather(make_src(), make_idx(2), make_dst()));
}

// Negative index: pre-fix wraps to a huge size_t offset -> OOB read (ASan).
// Post-fix: rejected with ov::Exception.
TEST(NpuwUtilGatherBounds, NegativeIndexRejected) {
    EXPECT_THROW(ov::npuw::util::gather(make_src(), make_idx(-1), make_dst()), ov::Exception);
}

// Index >= src_shape[0]: pre-fix reads past the src allocation (ASan).
// Post-fix: rejected with ov::Exception.
TEST(NpuwUtilGatherBounds, OverflowIndexRejected) {
    EXPECT_THROW(ov::npuw::util::gather(make_src(), make_idx(4), make_dst()), ov::Exception);
}

}  // namespace
