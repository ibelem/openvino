// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for openvino/src/plugins/intel_cpu/src/nodes/range.cpp:138-143
// (Range::getWorkAmount). Pre-fix: i32 Range with start=0, limit=INT_MIN, step=1
// triggers signed-overflow UB on `-iSpan` and a div_up result that casts to an
// enormous size_t work_amount -> redefineOutputMemory huge allocation (ASan/OOM).
// Post-fix: int64 magnitude computation + zero-step guard must reject/clamp so no
// absurd output extent is produced.
//
// SKELETON: exact single-op Range test fixture symbols must be copied from the
// intel_cpu single-layer test tree before this compiles.
#include <gtest/gtest.h>
// TODO: include the correct headers, e.g.
//   #include "single_layer_tests/classes/range.hpp" or the ngraph op + CPU
//   reference infra under src/plugins/intel_cpu/tests/unit/ and shared_test_classes.

TEST(CpuRangeNode, IntMinSpanDoesNotOverflowWorkAmount) {
    // TODO: build an i32 Range op with constant inputs start=0, limit=INT_MIN, step=1
    //   using the same builder helper the existing Range tests use
    //   (ov::op::v4::Range or the CPU node wrapper).
    // TODO: compile for CPU as a DYNAMIC-shape model so redefineOutputMemory() path
    //   (range.cpp:153-155) is exercised.
    // EXPECTATION (post-fix): inference either throws a bounded ov::Exception or
    //   produces a sane (non-astronomical) output extent; it must NOT attempt a
    //   ~2^31/2^63 element allocation.
    // ASSERT_NO_FATAL_FAILURE(infer());  // pre-fix: ASan signed-overflow / bad_alloc
    // EXPECT_LT(output_element_count, kSaneUpperBound);
    GTEST_SKIP() << "TODO: wire up intel_cpu Range single-op fixture symbols";
}
