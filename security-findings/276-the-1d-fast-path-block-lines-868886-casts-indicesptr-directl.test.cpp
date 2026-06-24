// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for OOB write at
// targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883
// The 1D i32 fast-path writes pdst[pindices[i]] with no bounds check.
// Pre-fix: an out-of-range index (negative or >= srcLength) triggers an
// ASan heap-buffer-overflow on the dst tensor write.
// Post-fix: ScatterUpdate::execute must reject the index (CPU_NODE_ASSERT ->
// ov::Exception), so the infer call throws instead of corrupting memory.
//
// Target: ov_cpu_unit_tests (intel_cpu). The exact subgraph-test helper symbols
// must be confirmed against intel_cpu/tests/unit/ before use.

#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test harness headers used by existing
// node tests under src/plugins/intel_cpu/tests/unit/ (e.g. the ov::Model /
// ov::op::v3::ScatterUpdate builder + a compiled infer-request helper).

TEST(ScatterUpdate1DFastPath, OutOfRangeIndexIsRejected) {
    // Build the exact fast-path shape: DATA 1D i32 len<=64, INDICES 1D i32,
    // UPDATE matching INDICES, AXIS = 0.
    const std::vector<int32_t> data(8, 0);          // srcLength = 8 (<=64)
    const std::vector<int32_t> indices = {10000};   // far past srcLength -> OOB
    const std::vector<int32_t> updates = {42};
    const std::vector<int64_t> axis = {0};

    // TODO: construct ov::op::v3::ScatterUpdate(data, indices, updates, axis),
    // wrap in ov::Model, core.compile_model(model, "CPU"), create infer request,
    // set the input tensors above.

    // The unchecked write at scatter_update.cpp:883 must now be guarded.
    // EXPECT_THROW(infer_request.infer(), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu subgraph/infer harness; assert "
                    "EXPECT_THROW(infer(), ov::Exception) for OOB index.";
}
