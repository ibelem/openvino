# Security finding #154: At line 603, `inNum = edge->getInputNum()` returns the raw `parent_…

**Summary:** At line 603, `inNum = edge->getInputNum()` returns the raw `parent_…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker-controlled graph with a parent node whose `outConfs` has fewer entries than the edge's `parent_port` value (e.g., `parent_port = 5` but `outConfs.size() == 1`) triggers an out-of-bounds read on a heap-allocated `std::vector<PortConfig>`, potentially leaking heap memory contents or, with a sufficiently crafted heap layout, causing a process crash (DoS). Exploitable during model loading in any application using the OpenVINO CPU plugin.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/node.cpp:603` — `Node::getParentOutputMemDesc()`
**Validated for repos:** openvino
**Trust boundary:** External ONNX/OpenVINO IR graph topology → EP graph builder → CPU plugin Node/Edge construction; `parent_port` in the Edge is set directly from the graph's port index without a range check

## Description / Root cause
At line 603, `inNum = edge->getInputNum()` returns the raw `parent_port` integer stored in the Edge (edge.cpp:292-294). Line 601 asserts `parentOutConfs` is non-empty, but there is no check that `inNum >= 0 && inNum < (int)parentOutConfs.size()` before the vector subscript at line 605 (`parentSpd->getConfig().outConfs[inNum].getMemDesc()`). Compare: the sibling functions `selectPreferPrimitiveDescriptor` (node.cpp:301-304), `selectPreferPrimitiveDescriptorWithShape` (node.cpp:396-399), and `Edge::getInputPortDesc` (edge.cpp:393-395) all clamp `inNum` to 0 when it is out of range, but `getParentOutputMemDesc` omits this guard entirely.

**Validator analysis:** Confirmed by reading the cited code. getParentOutputMemDesc (node.cpp:595-606) checks only parentSpd!=null and !parentOutConfs.empty(); it then subscripts outConfs[inNum] where inNum is the raw parent_port returned verbatim by Edge::getInputNum() (edge.cpp:292-294). Three siblings in the same tree clamp this exact value when out of range — selectPreferPrimitiveDescriptor (node.cpp:302-304), selectPreferPrimitiveDescriptorWithShape (node.cpp:397-399), and Edge::getInputPortDesc (edge.cpp:393-395) — which is direct evidence the OV developers treat inNum>=outConfs.size() (and inNum<0) as a real, occurring condition (e.g. nodes whose selected PD has fewer outConfs than output ports). The CWE-125 (OOB read) classification is accurate: std::vector::operator[] performs no bounds check, and a negative inNum converts to a huge size_type, widening the OOB. Impact (heap OOB read → info leak / DoS during model compile) is plausible though it requires a node whose selected PD's outConfs is smaller than the referenced parent_port; ov::Model validation generally keeps port indices within a node's output count, so exploitability is conditional on that PD/output mismatch — which is precisely why the siblings clamp. The proposed fix (clamp inNum to 0 when <0 or >=parentOutConfs.size(), matching the established pattern) is correct and sufficient; an OPENVINO_THROW is an acceptable stricter alternative. Verdict for openvinoEp is rejected: the defect is real but its code and the parent_port assignment are not authored in the EP, and the stated EP-graph-builder path is incorrect.

## Exploit / Proof of Concept
Craft an ONNX or OpenVINO IR model where a node has an edge with `parent_port` set to an index ≥ the number of output configurations on the parent node's selected primitive descriptor. When the CPU plugin invokes `getParentOutputMemDesc` (e.g., during `initOptimalPrimitiveDescriptor` or reorder-insertion passes), it subscripts `outConfs[inNum]` with an out-of-range index, reading memory beyond the vector's buffer. A negative `parent_port` (inNum < 0) also bypasses the only check (non-empty) and causes a large-index OOB read due to unsigned wrap in the vector subscript operator.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_cpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter=NodeGetParentOutputMemDesc.ParentPortOutOfRangeIsClamped . Expected pre-fix: ASan 'heap-buffer-overflow READ' inside Node::getParentOutputMemDesc at node.cpp:605 (std::vector<PortConfig>::operator[]). Post-fix (inNum clamped to 0): test passes / no ASan report. Note: skeleton — fill in the Node/Edge fixtures (TODOs) before it will compile.

## Suggested fix
Apply the same guard already present in `selectPreferPrimitiveDescriptor` (lines 301-304) and `Edge::getInputPortDesc` (lines 393-395). In `getParentOutputMemDesc`, after line 603, add:
```cpp
if (inNum < 0 || static_cast<size_t>(inNum) >= parentOutConfs.size()) {
    inNum = 0;
}
```
or alternatively throw a descriptive `OPENVINO_THROW` if an out-of-range port is a hard error. This is consistent with the established pattern across the same file.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #154.
