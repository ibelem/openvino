// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for OOB read at openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
// (Gather::exec1DCase reads psrc[ii] with no bounds check on the runtime-supplied
//  GATHER_INDICES value). The assertion: with a 1-D i32 data tensor of 5 elements and
//  an out-of-range index (65, or -1 with reverseIndexing=false, or a strongly negative
//  index with reverseIndexing=true), inference must NOT read out of bounds — under ASan
//  the pre-fix code traps; the post-fix code returns 0 for the offending position.
//
// TODO: This is a SKELETON. The CPU plugin unit harness (ov_cpu_unit_tests) builds Gather
//   subgraph tests via ov::test::SubgraphBaseTest / ngraph ops. Fill in the exact include
//   paths and helper names by reading src/plugins/intel_cpu/tests/unit/ before use.
#include <gtest/gtest.h>
// TODO: #include the CPU node test fixtures, e.g. "nodes/..." and openvino/op/gather.hpp

TEST(GatherExec1DCaseOOB, RejectOutOfRangeIndex) {
    // TODO: build a Gather op:
    //   data  = Constant<i32>{shape={5}, values={10,11,12,13,14}}  (dataSrcRank==1, dims<=64 -> canOptimize1DCase)
    //   axis  = 0
    //   indices = Parameter<i32>{shape={1}}  // attacker-controlled GATHER_INDICES
    // Compile for CPU, then run inference three times with indices = {65}, {-1}, {-100}.
    //
    // Pre-fix: exec1DCase() dereferences psrc[ii] OOB -> ASan heap-buffer-overflow.
    // Post-fix: out-of-range indices produce 0 in the output (matching execReference()).
    //
    // EXPECT_NO_THROW(infer_with_index(65));   // and no ASan abort
    // EXPECT_EQ(out_value, 0);
    GTEST_SKIP() << "TODO: wire up CPU Gather subgraph fixture (see intel_cpu/tests/unit/)";
}