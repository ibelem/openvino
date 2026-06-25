// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the OOB read at
//   openvino/src/core/reference/src/op/concat.cpp:58-61
// Pre-fix: op::v0::Concat::evaluate on i4 inputs with steps>1 makes
// copy_elements read arg+step*(N/steps) bytes from an N/2-byte packed
// buffer (heap OOB read; flagged by ASan). Post-fix: in_offset is a true
// byte offset and the read stays in bounds, so evaluate succeeds and the
// concatenated result is correct.
//
// TODO: confirm exact i4 tensor/Constant construction API and the test
//       target/include paths against the existing core reference/op test
//       tree (e.g. src/core/tests/eval.cpp) before building.
#include <gtest/gtest.h>
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(eval_concat, i4_axis_gt0_no_oob_read) {
    // Two i4 inputs shaped [2,4] (N=8 elems = 4 packed bytes each),
    // concatenated on axis=1 -> out [2,8], steps = out_shape[0] = 2.
    const Shape in_shape{2, 4};
    // TODO: fill with valid packed i4 data; data() byte size must be 4.
    auto a = op::v0::Constant::create(element::i4, in_shape, std::vector<int>(8, 1));
    auto b = op::v0::Constant::create(element::i4, in_shape, std::vector<int>(8, 2));
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{a, b}, /*axis=*/1);

    Tensor out(element::i4, Shape{2, 8});
    TensorVector outputs{out};
    TensorVector inputs{a->get_tensor_view(), b->get_tensor_view()}; // TODO: adjust accessor
    // Pre-fix this evaluate() reads past the 4-byte i4 source buffers.
    ASSERT_TRUE(concat->evaluate(outputs, inputs));
}
