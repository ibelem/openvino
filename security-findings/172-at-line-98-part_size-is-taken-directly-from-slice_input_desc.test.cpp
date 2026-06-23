// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 in op::v0::TensorIterator::validate_and_infer_types.
// Encodes the fix for:
//   targets/openvino/src/core/src/op/tensor_iterator.cpp:98
//     m_num_iterations = (std::abs(end - start) + 1) / part_size;
// where part_size == slice_input_description->m_part_size (default 0,
//   targets/openvino/src/core/include/openvino/op/util/multi_subgraph_base.hpp:99).
//
// Pre-fix: a SliceInputDescription with part_size == 0 reaches the division at
//   line 98 and triggers an integer divide-by-zero (SIGFPE / FPE_INTDIV abort).
// Post-fix: a NODE_VALIDATION_CHECK rejects part_size == 0, raising
//   ov::NodeValidationFailure (subclass of ov::Exception) instead of crashing.
//
// Style mirrors src/core/tests/type_prop/tensor_iterator.cpp.

#include "openvino/op/tensor_iterator.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, tensor_iterator_slice_input_zero_part_size_throws) {
    // Outer sliced input.
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{32, 40, 10});

    // Trivial body: Xi -> Add(Xi, Xi) -> Zo
    auto Xi = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = make_shared<op::v1::Add>(Xi, Xi);
    auto body = make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi});

    auto ti = make_shared<op::v0::TensorIterator>();
    ti->set_body(body);

    // set_sliced_input(body_parameter, value, start, stride, part_size, end, axis)
    // part_size == 0 is the crafted malicious value (the divisor at line 98).
    ti->set_sliced_input(Xi, X, /*start=*/0, /*stride=*/0, /*part_size=*/0, /*end=*/-1, /*axis=*/1);
    ti->get_iter_value(Zo, -1);

    // Pre-fix this aborts with SIGFPE inside validate_and_infer_types (line 98).
    // Post-fix the NODE_VALIDATION_CHECK turns it into a catchable exception.
    EXPECT_THROW(ti->validate_and_infer_types(), ov::Exception);
}