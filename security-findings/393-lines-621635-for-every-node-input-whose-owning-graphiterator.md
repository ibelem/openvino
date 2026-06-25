# Security finding #393: Lines 621–635: for every node input whose owning `GraphIteratorProt…

**Summary:** Lines 621–635: for every node input whose owning `GraphIteratorProt…

**CWE IDs:** CWE-400: Uncontrolled Resource Consumption
**Severity / Impact:** CPU-time and heap-memory denial of service. A crafted ONNX subgraph (e.g. 50 000 nodes each referencing a single parent-scope tensor) causes quadratic wall-clock time and O(N·K) redundant `shared_ptr` copies in `m_decoders`, exhausting memory and/or stalling the process. Affects any application that loads attacker-controlled ONNX models through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:621` — `GraphIteratorProto::reset()()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied ONNX model file / in-memory ModelProto ingested via GraphIteratorProto::initialize()

## Description / Root cause
Lines 621–635: for every node input whose owning `GraphIteratorProto` is a parent scope (`tensor_owner != this`), the code executes `m_decoders.insert(m_decoders.begin() + top_index, decoder_proto_tensor)` (line 633) and increments `top_index`. There is (a) no deduplication check — the same parent-scope tensor is re-inserted once per referencing input across all nodes — and (b) no cap on `top_index` or the number of such inserts. `std::vector::insert` at an interior position costs O(current_size) because it must shift all subsequent elements. With N subgraph nodes each carrying K outer-scope inputs, total insert cost is O((N·K)²). The `reserve()` at line 572 covers only `initializer_size + input_size + output_size + node_size` — it provides no headroom for these injected parent-tensor entries — so every batch of inserts beyond that reservation also triggers a heap reallocation, amplifying the cost further.

**Validator analysis:** The defect is real in the OpenVINO ONNX frontend. In reset() the node loop (621-635) calls m_decoders.insert(m_decoders.begin()+top_index, decoder) for every node input whose owning iterator is a parent scope (tensor_owner!=this). get_tensor() (542-558) recurses to m_parent on a cache miss, so subgraph nodes referencing outer-scope tensors take this branch. Each interior insert shifts (size-top_index) elements, and since node-output decoders are continually push_back'd at the tail (652) the shifted tail grows ~linearly with node count, so N such inserts cost O(N^2) shifts; the reserve() at 572 does not account for these injected entries. There is no deduplication, so referencing the same parent tensor from N nodes inserts N redundant shared_ptr copies. This matches CWE-400 (algorithmic/CPU DoS); the 'memory exhaustion' claim is weaker — memory is O(N) (linear, redundant), not unbounded — the dominant harm is quadratic CPU time. EP-boundary (openvinoEp) reachability is not demonstrated by the finding (the modern plugin_impl/ORT-ABI EP is not shown to feed subgraphs through the proto-based ONNX frontend), so I reject that repo on unproven reachability per the skeptical standard. The proposed fix is correct and sufficient: the deduplication shadow-set is the essential change (eliminates the redundant inserts and the worst quadratic factor); the cleanest form is the suggested two-pass — collect distinct parent-scope decoders, then perform a single batched insert at position 0 — reducing to O(N*K). A hard cap is a reasonable defense-in-depth but secondary. Note: a true regression test for an algorithmic DoS must assert on the post-fix structural invariant (no duplicate parent decoders in m_decoders) rather than wall-clock time, and requires a crafted ONNX Loop/If model fixture — hence a skeleton.

## Exploit / Proof of Concept
Craft an ONNX model containing a subgraph operator (e.g. `Loop` or `If`) whose body graph has N nodes (e.g. 50 000 `Identity` nodes), each with one input referencing a tensor defined in the outer (parent) graph scope. Per the ONNX spec this is fully valid and passes schema validation. When the frontend calls `GraphIteratorProto::reset()` on the subgraph iterator, the inner loop at lines 628–635 executes N inserts at positions 0, 1, …, N-1 of `m_decoders`. Each `insert` at position i shifts (current_size − i) elements rightward; summed over all N inserts the total element-shift count grows as O(N²). For N = 50 000 that is ~2.5 × 10⁹ element moves. No upstream guard in `initialize()` (lines 531–540) or `get_tensor()` (lines 542–558) prevents this: `get_tensor()` simply recurses to the parent on cache miss and returns the parent's `shared_ptr`, and the only early-return in `reset()` (line 569) fires only when `m_decoders` is already non-empty, not on the first call.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for graph_iterator_proto.cpp:621-635 (CWE-400 quadratic insert / no dedup).
// Pre-fix: a subgraph whose N nodes each reference the SAME outer-scope tensor causes
//   N redundant m_decoders.insert(begin()+top_index,...) at lines 633-634 (O(N^2) shifts,
//   O(N) duplicate shared_ptr copies). Post-fix (dedup): the parent tensor is injected once.
// This test loads a crafted model and asserts the model imports correctly and quickly;
// the structural invariant (no duplicate parent-decoder injection) is what the fix encodes.
#include "onnx_utils.hpp"   // TODO: confirm helper header for convert_model in ov_onnx_frontend_tests
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted fixture models/subgraph_outer_scope_fanout.onnx with a Loop/If
//       body containing many (e.g. 20000) Identity nodes that all read ONE outer-scope tensor.
//       Per-ONNX-spec valid; passes schema validation; triggers the parent-scope insert branch.
TEST(onnx_importer, subgraph_outer_scope_fanout_no_quadratic_insert) {
    std::shared_ptr<ov::Model> model;
    // Must complete (no hang / no OOM) and import successfully once the insert path is deduped/batched.
    OV_ASSERT_NO_THROW(model = convert_model("subgraph_outer_scope_fanout.onnx"));
    ASSERT_NE(model, nullptr);
    // TODO: if a white-box hook is available, assert m_decoders contains the parent tensor
    //       exactly once (not once per referencing node).
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan/leak coverage). Run: ov_onnx_frontend_tests --gtest_filter=onnx_importer.subgraph_outer_scope_fanout_no_quadratic_insert . Pre-fix symptom: test wall-clock blows up (quadratic) or the process is killed for OOM/timeout under a large-N fixture; post-fix it completes promptly. Requires crafting the subgraph_outer_scope_fanout.onnx fixture (TODO in test) since the trigger needs a binary ONNX model with a Loop/If body referencing an outer-scope tensor from many nodes.

## Suggested fix
Before executing the insert, check whether `decoder_proto_tensor` is already present among the first `top_index` entries of `m_decoders` (use a `std::unordered_set<std::shared_ptr<DecoderProtoTensor>>` shadow set keyed by pointer or tensor name, populated alongside `top_index`). Insert only on a cache miss. Additionally, consider replacing the mid-vector insert pattern with a two-pass strategy: collect parent-scope decoders in a separate `std::vector`, deduplicate, then `std::vector::insert` the whole batch once at position 0 before appending node decoders — reducing total complexity from O((N·K)²) to O(N·K). Also add a hard limit on the total number of parent-tensor injections (e.g. refuse if `top_index` would exceed a configurable cap such as `1 << 20`) to guard against models that legitimately but excessively reference outer-scope tensors.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #393.
