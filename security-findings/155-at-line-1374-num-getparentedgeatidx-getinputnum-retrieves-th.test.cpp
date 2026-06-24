// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression for the missing upper-bound check at
// openvino/src/plugins/intel_cpu/src/node.cpp:1374-1385
// (Node::getConsistentInputDesc indexes parentSelectedPD->getConfig().outConfs[num]
//  after only checking num >= 0, unlike the clamps at node.cpp:302-304,
//  edge.cpp:393-395 / 415-417). Pre-fix: ASan heap-buffer-overflow read when a
//  parent edge's parent_port >= parent node's selected outConfs.size().
//
// SKELETON: the exact graph/Node fixtures and helper symbols in
// src/plugins/intel_cpu/tests/unit/ must be confirmed by reading that tree.
#include <gtest/gtest.h>
// TODO: include the real CPU-plugin unit-test helpers, e.g.
//   #include "nodes/node_config.h"
//   #include "graph.h" / "node.h" / "edge.h"
// and whatever dummy-Node factory the existing ov_cpu_unit_tests use.

using namespace ov::intel_cpu;

TEST(NodeConsistentInputDesc, ParentPortOutOfRangeDoesNotReadOOB) {
    // TODO: build a parent Node whose selected PrimitiveDescriptor config has
    //       outConfs.size() == 1, and a child Node connected by an Edge whose
    //       parent_port == 5 (>=0 but >= outConfs.size()).
    //   auto parent = makeDummyNode(/*outConfs=*/1);
    //   auto child  = makeDummyNode(/*inConfs=*/1);
    //   auto edge   = std::make_shared<Edge>(parent, child, /*pr_port=*/5, /*ch_port=*/0);
    //   wireEdge(parent, child, edge);
    //
    // Pre-fix: getConsistentInputDesc indexes outConfs[5] -> ASan OOB read.
    // Post-fix (clamp to 0 / bound check): returns a valid desc, no OOB.
    //
    //   const auto& cfg = child->getSelectedPrimitiveDescriptor()->getConfig();
    //   EXPECT_NO_THROW({ auto pd = child->getConsistentInputDesc(cfg, 0); (void)pd; });
    GTEST_SKIP() << "TODO: provide intel_cpu Node/Edge fixtures to drive getConsistentInputDesc with parent_port >= outConfs.size().";
}