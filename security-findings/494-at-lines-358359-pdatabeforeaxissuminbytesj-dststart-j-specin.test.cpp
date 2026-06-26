// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for gather.cpp:358-359 (and 354-357): byte offsets computed as
// uint64_t must not be truncated into std::vector<int> (gather.h:39-43). Pre-fix, a
// Gather whose data buffer exceeds INT32_MAX bytes yields wrapped/negative offsets and
// an OOB read in the JIT kernel (ASan heap-buffer-overflow). Post-fix, the node must
// either use 64-bit offsets or reject the oversized stride in prepareParams.
//
// NOTE: a real trigger needs a >2GB source tensor, which is not feasible to allocate in
// a normal CI unit test. This is therefore a SKELETON; fill in the TODOs to run it on a
// large-memory host, or convert it into a guard-assertion test once prepareParams adds
// the CPU_NODE_ASSERT recommended in the fix.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node test helpers used by other ov_cpu_unit_tests
//       (e.g. the Graph/Node single-layer test fixture under
//        src/plugins/intel_cpu/tests/unit/). Read that tree for exact symbol names.

TEST(GatherCpuNode, OffsetStrideDoesNotTruncateToInt32) {
    // TODO: build an ov::op::v8::Gather with a static data shape whose
    //       axisDim * afterAxisSize * dataTypeSize > INT32_MAX so that
    //       axisAndAfterAxisSizeInBytes overflows 32 bits.
    //   const ov::Shape dataShape = { /* beforeAxis */ 2, /* axisDim */ 0x20000001, 1 };
    //   const ov::Shape idxShape  = { 2 };
    //   const int64_t axis = 1;
    // TODO: compile the subgraph for the CPU plugin and invoke prepareParams/createPrimitive.
    //
    // Expected behaviour AFTER the fix (pick one, matching the chosen remediation):
    //   (a) guard variant: createPrimitive/prepareParams rejects the oversized stride.
    //       EXPECT_THROW(infer_request.infer(), ov::Exception);
    //   (b) 64-bit variant: offsets remain correct; no ASan heap-buffer-overflow and
    //       results match the reference Gather.
    //
    // Pre-fix, variant (b) run under ASan reports a heap-buffer-overflow READ inside
    // jitUniGatherKernel due to the negative wrapped dataBeforeAxisSumInBytes offset.
    GTEST_SKIP() << "Skeleton: requires >2GB data tensor or the prepareParams size guard; see TODOs.";
}
