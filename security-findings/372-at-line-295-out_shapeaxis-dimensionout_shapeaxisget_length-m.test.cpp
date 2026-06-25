// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 at openvino/src/core/src/op/loop.cpp:295 (multiply
// out_shape[axis].get_length() * m_num_iterations with no overflow guard) and the
// missing bounds check at loop.cpp:124 (m_num_iterations = val[0]).
// Pre-fix: validate_and_infer_types() computes a wrapped/negative concat-axis
// dimension (UB signed overflow) instead of rejecting the input.
// Post-fix: a non-negative-but-overflowing trip_count must be rejected via
// ov::Exception (NODE_VALIDATION_CHECK).
//
// Harness: ov_core_unit_tests, type_prop style (src/core/tests/type_prop/loop.cpp).
// TODO: confirm exact helper/include names by reading src/core/tests/type_prop/loop.cpp
//       and src/core/tests/type_prop/tensor_iterator.cpp before use.

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

TEST(type_prop, loop_concat_output_tripcount_overflow_is_rejected) {
    // ---- Build a trivial body: Xi (slice) + cond(true) ----
    auto Xi = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});
    auto cur_iter = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    // body execution condition folds to constant true -> condition_always_true
    auto body_cond = op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto body_out  = std::make_shared<op::v0::Result>(Xi);
    auto body_cond_res = std::make_shared<op::v0::Result>(body_cond);

    auto body = std::make_shared<Model>(ResultVector{body_out, body_cond_res},
                                        ParameterVector{cur_iter, Xi});

    // ---- Loop external inputs ----
    // trip_count chosen so slice_size(2) * trip_count overflows int64_t.
    const int64_t kEvilTrip = 4611686018427387904LL; // INT64_MAX/2 + 1
    auto trip_count = op::v0::Constant::create(element::i64, Shape{1}, {kEvilTrip});
    auto exec_cond  = op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto X          = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});

    auto loop = std::make_shared<op::v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(op::v5::Loop::SpecialBodyPorts{0, 1});
    loop->set_invariant_input(Xi, X);
    // ConcatOutputDescription along axis 0 -> triggers loop.cpp:295 multiply.
    loop->get_concatenated_slices(body_out, 0, 1, 1, -1, 0);

    // Pre-fix: validate_and_infer_types() silently wraps to a negative dim (UB).
    // Post-fix: the overflowing/over-large trip_count must be rejected.
    EXPECT_THROW(loop->validate_and_infer_types(), ov::Exception);
}