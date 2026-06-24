# Security finding #155: At line 1374, `num = getParentEdgeAt(idx)->getInputNum()` retrieves…

**Summary:** At line 1374, `num = getParentEdgeAt(idx)->getInputNum()` retrieves…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Same class as the flaw in `getParentOutputMemDesc`: heap out-of-bounds read on the parent node's `outConfs` vector, reachable during `initOptimalPrimitiveDescriptor` or `configureOutPortDesc` passes while loading a crafted model. Could cause crash (DoS) or heap memory disclosure.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/node.cpp:1374` — `Node::getConsistentInputDesc()`
**Validated for repos:** openvino
**Trust boundary:** External ONNX/OpenVINO IR graph topology → CPU plugin Node/Edge construction; `parent_port` in the Edge is set directly from graph port indices

## Description / Root cause
At line 1374, `num = getParentEdgeAt(idx)->getInputNum()` retrieves the raw `parent_port` from the edge. The guard at line 1375 only checks `num >= 0` but does NOT check `num < (int)parentSelectedPD->getConfig().outConfs.size()` before subscripting `outConfs[num]` at lines 1376 and 1385. A `parent_port` value that is non-negative but ≥ `outConfs.size()` bypasses the guard and causes an OOB read.

**Validator analysis:** CWE-125 OOB read is accurate. Edge::getInputNum() (edge.cpp:292-294) returns the raw parent_port with no clamping. getConsistentInputDesc (node.cpp:1374) reads outConfs[num] twice (1376, 1385) after only checking num>=0. The decisive evidence that num can be >= outConfs.size(): three sibling code paths operating on the exact same getInputNum()/outConfs pair defensively clamp out-of-range indices to 0 — selectPreferPrimitiveDescriptor (302-304), getInputPortDesc (393-395), and getOutputPortDesc (415-417). Those clamps would be dead code if num were guaranteed in range, so the unclamped access in getConsistentInputDesc is a genuine, reachable OOB read on the parent node's outConfs, matching the sibling flaw in getParentOutputMemDesc (source item 122). Impact (DoS / heap info disclosure during initOptimalPrimitiveDescriptor) is plausible. The proposed fix is correct in direction but should follow the established repo idiom (clamp to 0) rather than silently skip the block, since the function still must return a desc; either clamp `if (num < 0 || num >= (int)parentOutConfs.size()) num = 0;` before line 1376 (consistent with edge.cpp:393-395) or wrap as proposed. The proposed wrap-with-bound is sufficient to stop the OOB but changes behaviour to fall through to inConf desc when out of range, which is acceptable. Recommend the clamp-to-0 form for consistency with the rest of the file.

## Exploit / Proof of Concept
Craft a model where a parent node's selected primitive descriptor has N output configurations and an outgoing edge's `parent_port` is ≥ N but > 0 (so `num >= 0` passes). When the CPU plugin calls `getConsistentInputDesc` for the child node, `outConfs[num]` is subscripted out-of-bounds.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests, with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=NodeConsistentInputDesc.ParentPortOutOfRangeDoesNotReadOOB . Expected pre-fix: AddressSanitizer 'heap-buffer-overflow READ' inside Node::getConsistentInputDesc at node.cpp:1376 (std::vector outConfs operator[]). Expected post-fix: test passes with no ASan report.

## Suggested fix
Add an upper-bound check before line 1376:
```cpp
int num = getParentEdgeAt(idx)->getInputNum();
const auto& parentOutConfs = parentSelectedPD->getConfig().outConfs;
if (num >= 0 && static_cast<size_t>(num) < parentOutConfs.size()) {
    auto parentConf = parentOutConfs[num];
    // ... rest of existing logic ...
}
```
This matches the pattern used in `selectPreferPrimitiveDescriptor` (lines 300-304).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #155.
