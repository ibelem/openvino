// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/core/src/op/tensor_iterator.cpp:98
//     m_num_iterations = (std::abs(end - start) + 1) / part_size;
// where part_size == slice_input_description->m_part_size (line 87), default 0
// (multi_subgraph_base.hpp:99) and no guard exists between 87 and 98.
//
// Pre-fix: with a static-shaped sliced input and part_size==0 this performs an
//   integer divide-by-zero during validate_and_infer_types() -> SIGFPE (ASan/UBSan
//   reports "division by zero").
// Post-fix: the added NODE_VALIDATION_CHECK(this, part_size > 0, ...) converts it
//   into an ov::Exception (ov::NodeValidationFailure), which this test asserts.
//
// Mirrors the existing builder style in
//   openvino/src/core/tests/type_prop/tensor_iterator.cpp.

#include "openvino/op/tensor_iterator.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, tensor_iterator_slice_input_zero_part_size_throws) {
    // Static-shaped external input so both rank.is_static() (line 86) and
    // input_partial_shape[axis].is_static() (line 92) are satisfied, forcing
    // execution of the division at line 98.
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});

    auto Xi = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = make_shared<op::v0::Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = make_shared<op::v1::Multiply>(Xi, M_body);
    auto body = make_shared<ov::Model>(OutputVector{Zo}, ParameterVector{Xi, M_body});

    auto ti = make_shared<op::v0::TensorIterator>();
    ti->set_body(body);
    ti->set_invariant_input(M_body, M);

    // start=0, stride=2, part_size=0 (malicious), end=39, axis=1.
    // set_sliced_input() internally calls validate_and_infer_types(), which
    // reaches tensor_iterator.cpp:98 and divides by zero pre-fix.
    EXPECT_THROW(ti->set_sliced_input(Xi, X, /*start*/ 0, /*stride*/ 2,
                                      /*part_size*/ 0, /*end*/ 39, /*axis*/ 1),
                 ov::Exception);
}