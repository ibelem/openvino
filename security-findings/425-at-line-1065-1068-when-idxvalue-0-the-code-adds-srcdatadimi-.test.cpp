// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing per-component index bounds check in
// ScatterUpdate::scatterNDUpdate (ReduceNone) at
// targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1063-1079.
//
// Pre-fix: a ScatterNDUpdate with an out-of-range / wrap-inducing index value
// produces an unaligned dstOffset that passes the dstOffset<elementsCount guard
// (line 1073) yet drives a cpu_memcpy of srcBlockND[k]*dataSize bytes (line 1079)
// past the end of the data buffer -> heap OOB write (caught by ASan).
// Post-fix: the per-component assert (idxValue in [0,srcDataDim[i])) rejects the
// model, so inference throws ov::Exception instead of writing OOB.
//
// TODO: confirm exact target/helpers by reading
//   targets/openvino/src/plugins/intel_cpu/tests/unit/  (target: ov_cpu_unit_tests)
// and the ScatterNDUpdate single-layer test under
//   .../tests/functional/single_layer_tests/scatter_ND_update.cpp
// TODO: pick a data shape whose inner block srcBlockND[k] is NOT a power of two
//   (e.g. data shape {4,3}, indices last-dim k=1 -> srcBlockND[1]=3) and an
//   indices value crafted so that (uint64_t)idxValue*3 wraps to elementsCount-1.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node/graph test fixtures used by ov_cpu_unit_tests.

TEST(ScatterNDUpdate_ReduceNone, RejectsOutOfRangeIndexInsteadOfOobWrite) {
    // TODO: build an OV model: ScatterNDUpdate(data{4,3}, indices{1,1}, updates{1,3})
    //       with reduction == NONE.
    // TODO: set the single index component to a wrap-inducing int64 value V such
    //       that ((uint64_t)V * srcBlockND[1]) % 2^64 == elementsCount - 1 (= 11),
    //       i.e. an unaligned offset that still satisfies dstOffset < elementsCount.
    // TODO: compile for the CPU plugin and run inference inside the assertion.
    //
    // EXPECT_THROW(infer(model, inputs), ov::Exception);   // passes only after the
    //                                                       // per-component bounds fix
    GTEST_SKIP() << "Skeleton: fill in intel_cpu CPU-plugin fixture + crafted indices.";
}
