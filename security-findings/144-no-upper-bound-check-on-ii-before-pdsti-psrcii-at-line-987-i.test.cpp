// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read in Gather::exec1DCase()
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no upper-bound check on `ii` (only ii<0 handled, 980-986).
// Pre-fix: with a positive out-of-range index this reads past the i32 data buffer
//   (ASan: heap-buffer-overflow READ of size 4 in ov::intel_cpu::node::Gather::exec1DCase).
// Post-fix: index >= axisDim must be zero-filled (mirroring execReference() line 947/959-963),
//   so the output element for the OOB index is 0 and no OOB access occurs.
//
// Target harness: ov_cpu_unit_tests (gtest + ASan).
// NOTE: emitted as a SKELETON. The intel_cpu unit-test tree builds Gather via the
// internal graph/Node infra (see src/plugins/intel_cpu/tests/unit/ for the exact
// helpers/fixtures). Exact symbol names for constructing a single CPU Gather node
// with const data + const indices and invoking execute() were not verified read-only,
// so the construction steps below are TODOs.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

// TODO: include the intel_cpu test helpers actually used under
//       src/plugins/intel_cpu/tests/unit/ (graph builder / node fixtures).

TEST(GatherCpuNode1DCase, OutOfRangeIndexIsZeroFilledNotOOBRead) {
    // Data: 1-D i32 tensor, axisDim = 4  (<=64 so canOptimize1DCase path is selected,
    //       see gather.cpp:396-404). Keep it as a runtime input so Gather is NOT
    //       constant-folded and the CPU node actually executes.
    const std::vector<int32_t> data = {10, 11, 12, 13};
    // Indices: 1-D i32 tensor with a positive out-of-range value.
    const std::vector<int32_t> indices = {0x7FFFFFFF};

    // TODO: build a single CPU Gather node (axis=0, batchDims=0, reverseIndexing as default)
    //       with `data` on GATHER_DATA and `indices` on GATHER_INDICES, allocate the
    //       output, run prepareParams() + execute(). Use the intel_cpu unit-test graph
    //       fixture rather than the full plugin so exec1DCase() is exercised directly.
    // std::vector<int32_t> out(indices.size());
    // runGatherNode(data, {4}, indices, {1}, /*axis*/0, out);

    // Post-fix expectation: out-of-range index -> zero-fill (consistent with execReference).
    // EXPECT_EQ(out[0], 0);
    // Pre-fix: the line above never executes cleanly; ASan aborts on the OOB read at
    //          gather.cpp:987 before reaching here.
    GTEST_SKIP() << "TODO: wire up intel_cpu Gather node construction helpers from "
                    "src/plugins/intel_cpu/tests/unit/ to drive exec1DCase().";
}
