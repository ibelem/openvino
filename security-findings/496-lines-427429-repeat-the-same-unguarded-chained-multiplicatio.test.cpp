// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 narrowing at gather.cpp:418/427 (axisDim is `int` in gather.h:92
// but assigned from size_t getStaticDims()[axis]). Pre-fix, an axis dimension > INT_MAX
// truncates/sign-wraps and corrupts srcAfterBatchSizeInBytes used as a JIT kernel stride
// (execute:493-494) -> OOB. Post-fix (axisDim widened to int64_t + overflow guard) the
// large dimension is represented faithfully / rejected and no OOB stride is produced.
//
// NOTE: a fully self-contained, compilable test is NOT achievable here: triggering the
// real overflow needs a Gather data tensor with an axis dimension > 2^31 elements
// (a multi-GB allocation), which a unit test cannot allocate. This is therefore a
// SKELETON. Exact target/symbols (ov_cpu_unit_tests, Gather node test fixtures) must be
// confirmed against the intel_cpu tests tree before use.

#include <gtest/gtest.h>
// TODO: include the intel_cpu Gather node header and the unit-test graph/infer fixtures
//       used by other ov_cpu_unit_tests (confirm exact paths under
//       openvino/src/plugins/intel_cpu/tests/unit/).

TEST(GatherIntegerOverflow, AxisDimNarrowingProducesNoOObStride) {
    // TODO: build a dynamic-shape Gather node (isDataShapeStat == false OR
    //       isAxisInputConst == false) so prepareParams takes the line 416-429 branch.
    // TODO: provide a data tensor whose axis static dim exceeds INT_MAX (e.g. (1LL<<31)).
    //       Because real allocation of >2GB is impractical, instead unit-test the arithmetic
    //       directly: feed dataDims[axis] = (1ULL<<31) into the computation and assert
    //       axisAndAfterAxisSizeInBytes / srcAfterBatchSizeInBytes equal the exact 64-bit
    //       product (no truncation/wrap), e.g.:
    //
    //       const uint64_t axisDim = (1ULL << 31);
    //       const uint64_t afterAxisSizeInBytes = 4; // f32, afterAxisSize==1
    //       const uint64_t expected = axisDim * afterAxisSizeInBytes; // 8589934592
    //       EXPECT_EQ(node->axisAndAfterAxisSizeInBytes(), expected);
    //
    // Pre-fix this fails because `int axisDim` truncates (1<<31) to INT_MIN, yielding a
    // wrong/huge uint64_t product; post-fix (int64_t axisDim) it matches `expected`.
    GTEST_SKIP() << "Skeleton: wire to ov_cpu_unit_tests Gather fixtures; see TODOs.";
}
