// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for OOB read at intel_cpu/src/nodes/gather.cpp:987 (Gather::exec1DCase).
// Pre-fix: ASan reports heap-buffer-overflow READ when a 1-D i32 source (<=64 elems)
// is gathered with an out-of-range index (e.g. -10 with reverseIndexing, or 100).
// Post-fix (guard `if (static_cast<size_t>(ii) >= axisDim) { pdst[i]=0; continue; }`):
// out-of-range lanes are zero-filled and no OOB occurs.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). The 1-D fast path requires
// dataSrcRank<=1, i32 precision, dataDims[0]<=64, idxDims[0]<=64 (gather.cpp:396-403).
#include <gtest/gtest.h>
// TODO: include the intel_cpu node test fixtures used by intel_cpu/tests/unit
//       (e.g. the Graph/Node test helpers) — confirm exact headers/targets by reading
//       intel_cpu/tests/unit/CMakeLists.txt and an existing node unit test.

// TODO: Build a minimal CPU graph containing a single Gather node:
//   - GATHER_DATA: shape [5], precision i32, values {0,1,2,3,4}
//   - GATHER_INDICES: shape [1], precision i32, value {-10} (reverseIndexing=true case)
//   - axis = 0
// so prepareParams() sets canOptimize1DCase=true and exec1DCase() runs.
TEST(GatherExec1DCase, OutOfRangeIndexIsBoundsChecked) {
    // TODO: construct node, allocate src/idx/dst memory via the unit-test memory helpers,
    //       set pidx[0] = -10, run execute(), and assert dst[0] == 0 (zero-fill semantics).
    //       The decisive signal pre-fix is the ASan heap-buffer-overflow on psrc[ii].
    GTEST_SKIP() << "TODO: wire up intel_cpu node fixture to drive Gather::exec1DCase";
}
