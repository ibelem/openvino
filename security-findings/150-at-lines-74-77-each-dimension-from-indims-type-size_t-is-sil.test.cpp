// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190/CWE-787 in
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:74-86,96
// where dim vars (IW/IH/IC/B) are `int` and `dstIndex = b*IC*IH*IW + ...`
// is computed in signed 32-bit arithmetic, so a product >= 2^31 wraps
// negative and `dst_data[dstIndex]` writes out of bounds.
//
// This assertion encodes the fix: after widening the index math to int64_t/
// size_t and adding an OPENVINO_ASSERT(B*IC*IH*IW <= INT64_MAX, ...), a
// ReorgYolo op whose total element count exceeds INT_MAX must be rejected
// (or processed without OOB write) instead of silently overflowing.
// Pre-fix: ASan reports a heap-buffer-overflow WRITE in ReorgYolo::execute.
// Post-fix: the op throws ov::Exception (or completes in-bounds).
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit.
//
// NOTE: a fully self-contained, *compilable* test that actually triggers the
// overflow is impractical — it requires ~2^31 output elements (~8 GB+ of
// allocation) for the multiply to cross INT_MAX. This is therefore emitted as
// a SKELETON; the real fix should add a cheap pre-loop range check that this
// test can exercise without giant allocations.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node test scaffolding actually used by the
// existing tests under openvino/src/plugins/intel_cpu/tests/unit/ — read that
// directory to learn the exact fixture headers (e.g. the node/graph test
// helpers) and the correct namespace for ReorgYolo / Graph construction.

TEST(ReorgYoloOverflow, RejectsIndexExceedingInt32) {
    // TODO: build a single-node ReorgYolo graph with f32 ncsp input whose
    //       static dims make B*IC*IH*IW exceed INT_MAX, e.g.
    //       inDims = {3, 32768, 32768, 1}, stride = 1.
    //       Use the intel_cpu unit-test graph/node builder (look up the exact
    //       API in tests/unit/ — do NOT guess symbol names).
    //
    // TODO: after the fix adds OPENVINO_ASSERT on the total element count,
    //       expect construction/prepareParams/execute to throw:
    //
    //   EXPECT_THROW(node->execute(stream), ov::Exception);
    //
    // Pre-fix this path silently computes a negative dstIndex and writes OOB,
    // which ASan flags as a heap-buffer-overflow WRITE in ReorgYolo::execute
    // (reorg_yolo.cpp:96).
    GTEST_SKIP() << "Skeleton: fill in intel_cpu node builder + post-fix range "
                    "check assertion; see reorg_yolo.cpp:74-96";
}
