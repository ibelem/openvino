# Security finding #480: Line 456 calls `m_parent_graph->get_ov_node_from_cache(name)` on a …

**Summary:** Line 456 calls `m_parent_graph->get_ov_node_from_cache(name)` on a …

**CWE IDs:** CWE-674: Uncontrolled Recursion
**Severity / Impact:** Stack exhaustion / crash (DoS). An attacker-supplied ONNX model with N levels of nested Loop/If subgraphs whose innermost body references a tensor from the outermost scope drives N recursive stack frames during model parsing. With no depth limit this reliably overflows the stack on all platforms, killing the inference-engine process.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph.cpp:456` — `Subgraph::get_ov_node_from_cache()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model subgraph attribute: node input names resolved via parent Graph pointer chain at model load time

## Description / Root cause
Line 456 calls `m_parent_graph->get_ov_node_from_cache(name)` on a raw `Graph*`. Because `m_parent_graph` can itself point to a `Subgraph`, the call dispatches virtually to `Subgraph::get_ov_node_from_cache`, which again calls its own `m_parent_graph->get_ov_node_from_cache` — a chain of recursive virtual calls with no depth counter, no iteration cap, and no cycle guard. The same pattern exists in `Subgraph::is_ov_node_in_cache` (line 449).

**Validator analysis:** The cited recursion is real: Subgraph::get_ov_node_from_cache (graph.cpp:452-469) and is_ov_node_in_cache (445-450) recurse up the m_parent_graph chain with no depth counter, iteration cap, or cycle guard. Because m_parent_graph is a Graph* that may itself be a Subgraph, the calls dispatch virtually and recurse to a depth equal to the model-controlled subgraph nesting level, which is reachable from untrusted ONNX model load. CWE-674 (Uncontrolled Recursion) and the DoS/stack-exhaustion impact are accurate, though the realistic crash requires hundreds-to-thousands of nested Loop/If bodies whose inner scope references an outer-scope tensor — and note that subgraph CONVERSION (Loop/If op translators recursing into bodies) already recurses to the same depth, so the cited line is one of several manifestations of the same root cause. The proposed fix (depth counter / iterative parent-chain walk) is correct for THIS function but is INCOMPLETE: it must also bound the recursive subgraph-conversion descent (a single shared MAX_SUBGRAPH_DEPTH checked at subgraph construction/convert time would cover both the cache lookups and the conversion recursion). I recommend converting the parent-chain lookups to an iterative loop (no stack growth) AND adding a global nesting-depth limit enforced when a Subgraph is constructed. I reject the EP repo because the vulnerable code and the cited model-load trust boundary are both inside openvino's onnx frontend, and propagation from the EP into that recursion with attacker-controlled nesting is not demonstrated.

## Exploit / Proof of Concept
Craft an ONNX model with N nested `Loop` operators where each loop body references a tensor name that exists only in the top-level graph scope. During `Subgraph::get_ov_node_from_cache(name)` at level N, the call chain traverses N `Subgraph` objects upward, each dispatching virtually to the next, until the stack overflows. N in the hundreds suffices; N=1000 is trivially encodable in protobuf.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-674 uncontrolled recursion in
// openvino/src/frontends/onnx/frontend/src/core/graph.cpp:456 (and :449).
// Pre-fix: importing a model with deeply nested Loop/If subgraphs whose inner
// body references an outermost-scope tensor recurses N frames through
// Subgraph::get_ov_node_from_cache and overflows the stack (ASan: stack-overflow).
// Post-fix: a MAX_SUBGRAPH_DEPTH guard makes convert_model reject the model with
// an ov::Exception instead of crashing.
//
// NOTE: this requires a crafted binary fixture (deeply_nested_loops.onnx) that is
// not present in the tree; emitted as a SKELETON.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: generate models/deeply_nested_loops.onnx with ~1000 nested Loop bodies,
//       each innermost body referencing a tensor defined only in the top-level
//       graph (forces Subgraph::get_ov_node_from_cache to walk all N parents).
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_deeply_nested_subgraph_recursion_guard) {
    EXPECT_THROW(convert_model("deeply_nested_loops.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_model_deeply_nested_subgraph_recursion_guard*. Pre-fix expected: ASan 'stack-overflow' / SEGV during convert_model while recursing Subgraph::get_ov_node_from_cache (graph.cpp:456). Post-fix expected: convert_model throws ov::Exception (depth-limit message) and the test passes. TODO: add the crafted deeply_nested_loops.onnx fixture under the onnx frontend test models dir.

## Suggested fix
Introduce a depth counter (or pass a `max_depth` parameter) through the recursive call. Before calling `m_parent_graph->get_ov_node_from_cache`, check `depth >= MAX_SUBGRAPH_DEPTH` (e.g. 64) and throw a descriptive `ov::Exception` or return an error Output. Alternatively, convert the parent-chain traversal to an iterative loop that walks the `Graph*` chain without recursion, eliminating stack growth entirely.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #480.
