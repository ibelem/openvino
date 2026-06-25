// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for CWE-190 at openvino/src/core/src/op/loop.cpp:295
// (unguarded signed int64_t `out_shape[axis].get_length() * m_num_iterations`).
// Pre-fix: with a huge TripCount constant the product overflows int64_t (UBSan: signed
// integer overflow) and the concat-axis dimension is silently clamped to 0 / wraps.
// Post-fix: the product is saturated (Dimension::dynamic or s_max), so the inferred
// concat-axis dimension is NOT a bogus small/zero static value.
//
// Harness: ov_core_unit_tests (gtest). Place in openvino/src/core/tests/type_prop/loop.cpp
// alongside the existing TYPED/TEST cases for op::v5::Loop.
//
// NOTE: building a full Loop (body model + SpecialBodyPorts + ConcatOutputDescription)
// requires the exact builder helpers used by the existing loop type_prop tests; the
// TODOs below mark the pieces to copy from those tests verbatim. Marked skeleton because
// the precise body-construction symbols were not read here.

#include <gtest/gtest.h>
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

TEST(type_prop_loop, trip_count_concat_axis_overflow_is_saturated) {
    // TODO: construct body Model exactly as existing loop type_prop tests do:
    //   - a body Parameter with a static rank-1 shape (e.g. {2}) feeding the concat output
    //   - a body condition Result that is Constant(true)  -> condition_always_true
    // TODO: build op::v5::Loop, set special body ports (body_condition_output_idx,
    //       current_iteration_input_idx) and get_concatenated_slices(...) for the body value
    //       along axis 0, matching the helper used in the existing tests.

    // Crafted TripCount so that dim(2) * trip_count overflows int64_t and (pre-fix) wraps:
    const int64_t kOverflowTrip = 0x4000000000000001LL;  // 2 * this overflows int64_t
    auto trip_count = op::v0::Constant::create(element::i64, Shape{}, {kOverflowTrip});
    auto exec_cond  = op::v0::Constant::create(element::boolean, Shape{}, {true});

    // TODO: auto loop = make_loop(trip_count, exec_cond, body, /*concat axis*/0, /*dim*/2);
    // loop->validate_and_infer_types();

    // EXPECTED post-fix: the concat-axis dimension must NOT be a wrapped/clamped small static
    // value. A correct saturating multiply yields a dynamic dimension (or s_max), so:
    //   const auto& out_ps = loop->get_output_partial_shape(/*concat out idx*/0);
    //   EXPECT_TRUE(out_ps[0].is_dynamic())
    //       << "loop.cpp:295 must saturate the trip_count*dim product, not overflow to 0";
    // Pre-fix this fails: out_ps[0] is a static 0 (or wrapped value) and UBSan reports
    // signed-integer-overflow at loop.cpp:295.
    GTEST_SKIP() << "skeleton: fill in body/Loop construction from existing loop type_prop tests";
}
