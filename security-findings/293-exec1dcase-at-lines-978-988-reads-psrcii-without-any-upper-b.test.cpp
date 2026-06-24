// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for OOB read at intel_cpu/src/nodes/gather.cpp:987 (exec1DCase).
// Pre-fix: psrc[ii] is read with ii==axisDim (negative idx, reverseIndexing=false)
// or ii>=axisDim (positive out-of-range idx) -> ASan heap-buffer-overflow read.
// Post-fix: out-of-range indices must yield 0 in the output, matching execReference
// (gather.cpp:947). Built for ov_cpu_unit_tests (subgraph/node test style).
//
// TODO(skeleton): confirm exact include paths + the intel_cpu node-test fixture
// helper (e.g. the GraphContext/Node build helpers under
// src/plugins/intel_cpu/tests/unit/). The 1D fast path requires:
//   - GATHER_DATA: 1-D i32 tensor, dim0 <= 64 (e.g. {1,2,3})
//   - GATHER_INDICES: 1-D i32 tensor, dim0 <= 64 with an OOB value (e.g. {99} or {-5})
//   - reverseIndexing == false on the Gather op
// so that prepareParams() sets canOptimize1DCase=true (gather.cpp:399-401) and
// exec1DCase() runs.
#include <gtest/gtest.h>
// TODO: include the intel_cpu node/graph unit-test harness headers used by the
// existing tests under src/plugins/intel_cpu/tests/unit/ (graph builder, infer).

TEST(GatherExec1DCase, OutOfRangeIndexIsZeroedNotOOBRead) {
    // TODO: build a Gather op with 1-D i32 data {1,2,3} (axisDim=3),
    //       1-D i32 indices {99} (positive OOB) and reverseIndexing=false.
    // TODO: run inference on the CPU plugin so Gather::exec1DCase executes.
    // Expected post-fix: output[0] == 0 (no OOB read). Pre-fix: ASan flags
    // heap-buffer-overflow at gather.cpp:987.
    // ASSERT_EQ(out[0], 0u);
    GTEST_SKIP() << "skeleton: wire up intel_cpu Gather node fixture";
}

TEST(GatherExec1DCase, NegativeIndexNoReverseIsZeroedNotOnePastEnd) {
    // TODO: data {1,2,3} (axisDim=3), indices {-5}, reverseIndexing=false.
    // Pre-fix: ii is set to axisDim=3 (gather.cpp:984) -> psrc[3] one-past-end read.
    // Expected post-fix: output[0] == 0.
    GTEST_SKIP() << "skeleton: wire up intel_cpu Gather node fixture";
}