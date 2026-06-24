// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the silent size_t->int narrowing at
// openvino/src/plugins/intel_cpu/src/nodes/conv.cpp:808-810 (Convolution::addFusedNode).
// Pre-fix: a fused depthwise-conv whose input/weight spatial static dim >= 2^31
// truncates `int src/krn/dst` (e.g. 2^31+5 -> 5), yielding a corrupted paddingR;
// post-fix (range-checked conversion / ptrdiff_t): the oversized dim is rejected
// with an ov::Exception instead of being silently wrapped.
//
// SKELETON — exact harness target + graph-builder helpers must be confirmed by
// reading the intel_cpu unit test tree (e.g. src/plugins/intel_cpu/tests/unit/).
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"

// TODO: include the correct intel_cpu unit-test fixture headers used to build a
//       Convolution node + a fusable depthwise Convolution and invoke addFusedNode.
//       The exact target is ov_cpu_unit_tests; symbol names below are placeholders.
TEST(ConvAddFusedNode, RejectsSpatialDimAboveIntMax) {
    // TODO: construct a Convolution node and a depthwise Convolution fusingNode
    //       whose input spatial static dim is (size_t)INT_MAX + 5.
    //       const size_t huge = static_cast<size_t>(std::numeric_limits<int>::max()) + 5;
    //
    // Pre-fix this silently wraps (huge -> 5) inside addFusedNode with no throw.
    // Post-fix the >INT_MAX dim must be rejected.
    // EXPECT_THROW(convNode->addFusedNode(dwConvFusingNode), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu Convolution fusion fixture";
}
