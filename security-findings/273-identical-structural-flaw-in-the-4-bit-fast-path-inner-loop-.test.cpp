// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression for OOB in Gather::execCompressed4Bit, gather.cpp:735-737.
// Encodes: a GatherCompressed with u4 data where afterAxisSize is NOT a
// multiple of scale_group_size must NOT read/write out of bounds (caught by
// ASan pre-fix; passes once the inner loop is clamped to srcIdx+afterAxisSize).
//
// SKELETON: exact GatherCompressed builder + tensor wiring must be filled in
// from the existing intel_cpu single-layer test helpers; symbol names below
// are placeholders pending a read of the surrounding test tree.
#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(GatherCompressed4Bit, NonDivisibleScaleGroupSizeNoOOB) {
    // TODO: build a GatherCompressed graph with:
    //   - GATHER_DATA: shape [vocab=4, afterAxisSize=7], precision u4
    //   - GATHER_SCALE: shape [4, 2]  -> scale_group_size = 7/2 = 3 (int trunc),
    //     i.e. afterAxisSize(7) % scale_group_size(3) != 0
    //   - GATHER_INDICES: {0}, axis const = 0  (drives the axis==0 fast path)
    //   - have_scalar_scale == false so cond3 is false and the cond1||cond2
    //     inner-loop branch at gather.cpp:730-741 is taken.
    // TODO: compile for the CPU plugin and infer; pre-fix this triggers an
    //       ASan heap-buffer-overflow at gather.cpp:736 (srcData[g>>1]) and
    //       gather.cpp:737 (pdst[dst_idx]).
    // TODO: assert inference completes and output buffer is untouched beyond
    //       afterAxisSize elements per row.
    GTEST_SKIP() << "Fill in GatherCompressed builder from intel_cpu test helpers";
}
