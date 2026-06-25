// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in Gather::exec1DCase (gather.cpp:987).
// Pre-fix: with a 1-D i32 data tensor of length N (<=64) and an index value
//   either >= N or < -N (reverseIndexing), psrc[ii] reads out of bounds
//   (caught by ASan). Post-fix: out-of-range indices yield 0 in the output
//   (mirroring Gather::execReference at gather.cpp:945-947).
//
// Harness: ov_cpu_unit_tests (intel_cpu component).
// TODO: confirm exact include paths / helper names against the existing
//       gather single-layer tests under
//       openvino/src/plugins/intel_cpu/tests/unit (or functional/single_layer_tests/gather.cpp).

#include <gtest/gtest.h>
// TODO: include the intel_cpu test scaffolding used by the existing Gather
//       node unit tests (e.g. the node test fixture header). Names below are
//       placeholders until verified by reading the surrounding test tree.

namespace ov {
namespace intel_cpu {
namespace node_test {

// TODO: replace with the project's actual Gather-node unit-test fixture.
TEST(GatherNode1DCase, OutOfRangeIndexDoesNotReadOOB) {
    // 1-D int32 data of length N=4 (<=64) -> triggers canOptimize1DCase
    // (prepareParams gather.cpp:399-401).
    const std::vector<int32_t> data = {10, 20, 30, 40};
    const int32_t axisDim = static_cast<int32_t>(data.size());

    // Indices that exercise both OOB paths:
    //   N+5  -> positive over-range (psrc[ii] past buffer)
    //   -(N+1) -> reverseIndexing under-range -> ii==-1 -> psrc[-1]
    const std::vector<int32_t> indices = {0, axisDim + 5, -(axisDim + 1)};

    std::vector<uint32_t> out(indices.size());

    // TODO: build and run the Gather node 1-D path with reverseIndexing=true
    //       using the intel_cpu node test fixture instead of this direct call.
    //       runGather1DCase(data, indices, /*reverseIndexing=*/true, out);

    // Post-fix expectation: in-range index returns the element; every
    // out-of-range index returns 0 (execReference semantics). Pre-fix this
    // test instead trips ASan (heap-buffer-overflow READ of 4 bytes).
    EXPECT_EQ(out[0], 10u);
    EXPECT_EQ(out[1], 0u);
    EXPECT_EQ(out[2], 0u);
}

}  // namespace node_test
}  // namespace intel_cpu
}  // namespace ov
