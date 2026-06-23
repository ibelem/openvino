// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for unchecked vector subscript in
//   openvino/src/plugins/intel_cpu/src/node.cpp:603-605
//   MemoryDescPtr Node::getParentOutputMemDesc(const EdgePtr& edge) {
//       ...
//       const int inNum = edge->getInputNum();          // line 603 (raw parent_port)
//       return parentSpd->getConfig().outConfs[inNum]... // line 605 (OOB if inNum >= outConfs.size())
//   }
// Encodes the fix: once getParentOutputMemDesc clamps inNum (as the siblings at
// node.cpp:302-304 / 397-399 / edge.cpp:393-395 do), calling it with an edge whose
// parent_port exceeds the parent SPD's outConfs.size() must NOT read out of bounds
// (returns outConfs[0]); pre-fix this trips ASan heap-buffer-overflow.
//
// Harness: ov_cpu_unit_tests (gtest + ASan), under src/plugins/intel_cpu/tests/unit/.
// SKELETON: building a Node with a selected primitive descriptor whose outConfs is
// smaller than the edge's parent_port, plus a connected parent Edge, needs internal
// CPU-plugin scaffolding whose exact symbols I could not confirm read-only.

#include <gtest/gtest.h>
// TODO: include the real headers used by the existing intel_cpu unit tests, e.g.
//   "nodes/..." , "edge.h", "node.h", "cpu_memory.h" — confirm exact paths under
//   src/plugins/intel_cpu/src and the test include dirs before use.

using namespace ov::intel_cpu;

TEST(NodeGetParentOutputMemDesc, ParentPortOutOfRangeIsClamped) {
    // TODO: construct a minimal parent Node and select a primitive descriptor whose
    //       getConfig().outConfs has size == 1 (non-empty, satisfying the assert at
    //       node.cpp:601).
    // TODO: construct a child Node and an Edge(parent, child, /*parent_port=*/5, 0)
    //       so that edge->getInputNum() == 5 (>> outConfs.size()).
    //   EdgePtr edge = std::make_shared<Edge>(parent, child, /*pr_port=*/5, /*ch_port=*/0);
    //
    // Pre-fix: getParentOutputMemDesc(edge) subscripts outConfs[5] -> ASan OOB read.
    // Post-fix: inNum is clamped to 0 and a valid MemoryDesc is returned.
    //
    //   EXPECT_NO_THROW({ auto md = child->getParentOutputMemDesc(edge); ASSERT_TRUE(md); });
    //
    // Also cover the negative-port path (parent_port = -1 -> size_type wrap -> huge OOB):
    //   EdgePtr negEdge = std::make_shared<Edge>(parent, child, /*pr_port=*/-1, 0);
    //   EXPECT_NO_THROW({ auto md = child->getParentOutputMemDesc(negEdge); ASSERT_TRUE(md); });
    GTEST_SKIP() << "TODO: provide intel_cpu Node/Edge fixtures with a selected PD whose "
                    "outConfs.size() < parent_port; see node.cpp:603-605.";
}
