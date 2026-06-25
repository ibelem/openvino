// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1107-1124
// (reduction overload of ScatterUpdate::scatterNDUpdate). Pre-fix: an out-of-range / negative
// indices tuple makes dstOffset wrap past the CPU_NODE_ASSERT at line 1117 and the reduction
// kernel at line 1124 writes OOB on the heap (ASan heap-buffer-overflow). Post-fix: the per-
// component range check throws, so the op is rejected before any write.
//
// TODO: target = ov_cpu_unit_tests (intel_cpu/tests/unit). Confirm exact fixture/helpers by
//       reading intel_cpu/tests/unit for an existing ScatterNDUpdate node single-layer test;
//       symbol names below are placeholders and MUST be replaced with the real harness API.
#include <gtest/gtest.h>
// TODO: include the real intel_cpu unit-test node-builder / ov::Model + ov::Core helpers.

TEST(scatter_nd_update_cpu, reduction_sum_rejects_out_of_range_indices) {
    // TODO: build an ov::Model with a v15 ScatterNDUpdate(reduction=SUM):
    //   data    : f32 shape {4}            (elementsCount = 4)
    //   indices : i64 shape {1,1}  value = {-1000000}  // residually negative -> wraps dstOffset
    //   updates : f32 shape {1}
    // TODO: compile on the CPU plugin and infer.
    // Expectation AFTER fix: per-component validation throws.
    // EXPECT_THROW(compiledModel.create_infer_request().infer(), ov::Exception);
    //
    // Pre-fix behaviour: no throw; ASan reports heap-buffer-overflow WRITE inside the
    // reduction kernel (scatter_update.cpp:1124).
    GTEST_SKIP() << "TODO: wire up ov_cpu_unit_tests ScatterNDUpdate fixture";
}