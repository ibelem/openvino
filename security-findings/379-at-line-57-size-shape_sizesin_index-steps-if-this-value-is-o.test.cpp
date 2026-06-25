// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for openvino/src/core/reference/src/op/concat.cpp:57-63
// (CWE-197 truncation -> CWE-908 uninitialized output byte for odd u4 per-step counts).
//
// Pre-fix: concat() silently copies floor(3/2)=1 byte per input and leaves the
//          trailing output byte unwritten -> ASan/MSan flags use-of-uninitialized,
//          and the result is wrong, but evaluate() does NOT throw.
// Post-fix: an even-count guard in reference::concat()/Concat::validate_and_infer_types()
//          rejects the non-byte-aligned u4 segment, so evaluate() throws.
//
// Lives alongside the core op_eval tests (target: ov_core_unit_tests, e.g. src/core/tests/eval.cpp).
// TODO: confirm exact include paths / test file location against the local core tests tree.
#include "gtest/gtest.h"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(eval_concat, u4_odd_per_step_count_must_be_rejected) {
    // Two u4 inputs of shape [3] concatenated on axis 0 -> per-step element count = 3 (odd).
    // 3 nibbles occupy 2 bytes; reference::concat() truncates 3/2 -> 1 byte (concat.cpp:57-60).
    auto a = std::make_shared<op::v0::Constant>(element::u4, Shape{3}, std::vector<uint8_t>{0, 0});
    auto b = std::make_shared<op::v0::Constant>(element::u4, Shape{3}, std::vector<uint8_t>{0, 0});
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{a, b}, 0);

    Tensor in_a(element::u4, Shape{3});
    Tensor in_b(element::u4, Shape{3});
    Tensor out(element::u4, Shape{6});  // 3 bytes; pre-fix byte[2] stays uninitialized
    TensorVector inputs{in_a, in_b};
    TensorVector outputs{out};

    // Once the even-count guard is added, the malformed u4 segment is rejected up-front.
    // Pre-fix this returns true and leaves out.data()[2] uninitialized (caught by ASan/MSan).
    EXPECT_ANY_THROW(concat->evaluate(outputs, inputs));
}
