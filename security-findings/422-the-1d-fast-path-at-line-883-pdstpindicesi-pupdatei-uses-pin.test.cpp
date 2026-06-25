// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-787 OOB write at
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:882-884
// The 1D fast-path writes pdst[pindices[i]] with no bounds check. A ScatterUpdate
// with data shape [1] (i32), indices=[-1] (or >=srcDataDim[0]) hits the fast-path
// (line 868) and writes out of bounds, returning at line 885 before any validation.
// Pre-fix: ASan heap-buffer-overflow on the dst allocation.
// Post-fix: index is range-checked and execution throws ov::Exception.
//
// TODO: place under openvino/src/plugins/intel_cpu/tests/unit/ and build with
//       ov_cpu_unit_tests. Confirm exact node/test helpers by reading the existing
//       intel_cpu unit test tree (e.g. nodes/ helpers) before use.
#include <gtest/gtest.h>
// TODO: include the intel_cpu node test fixtures / graph builder helpers used by
//       the existing ScatterUpdate unit tests (symbols unverified).

TEST(ScatterUpdateCpu, FastPathRejectsOutOfRangeIndex) {
    // TODO: build a ScatterUpdate node with:
    //   data    : i32, shape [1]  (srcDataDim[0] <= 64 -> enters fast path)
    //   indices : i32, shape [1], value = -1   (also test value = 64)
    //   update  : i32, shape [1]
    // and run execute().
    //
    // Expected after fix: ov::Exception (index out of range). Before fix this is
    // an undetected OOB write caught only by ASan.
    // EXPECT_THROW(run_scatter_update(/*data shape*/ {1}, /*indices*/ {-1}, /*update*/ {7}), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu ScatterUpdate node test fixture";
}