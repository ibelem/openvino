// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/core/src/op/constant.cpp:242-249
// (Constant::allocate_buffer): a Constant whose declared shape implies a
// byte size far larger than any backing data must be rejected with an
// ov::Exception instead of being passed uncapped to AlignedBuffer.
//
// Pre-fix: constructing the Constant attempts a multi-GB aligned_alloc
//          (succeeds and wastes memory, or throws std::bad_alloc, NOT ov::Exception).
// Post-fix: deserialization/allocation validates the declared size against the
//           available data and throws ov::Exception.
//
// NOTE: This is a SKELETON. The real trigger is deserialization of an IR where
// the declared shape exceeds the actual <value>/weights blob; constructing a
// Constant directly from a Shape does not model that mismatch and may simply
// allocate. The TODOs below name what is missing.

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"

using namespace ov;

TEST(constant, reject_excessive_declared_size_vs_data) {
    // TODO: replace direct construction with a read_model() of a crafted IR
    //       (.xml) whose <Const> declares element_type="f32" shape="2147483648"
    //       but supplies a tiny/empty backing buffer, so the fix can compare
    //       declared byte_size against the actual data size.
    const Shape excessive_shape{static_cast<size_t>(1) << 31};  // 2^31 f32 = 8 GB
    EXPECT_THROW(op::v0::Constant(element::f32, excessive_shape), ov::Exception);
}
