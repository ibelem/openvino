# Security finding #203: At line 551, `auto num_graphs = stream_graphs.size();` — if m_graph…

**Summary:** At line 551, `auto num_graphs = stream_graphs.size();` — if m_graph…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Denial of service: any attempt to run inference after model load with zero streams causes an immediate crash (SIGFPE or UB). Because the crash happens inside the inference hot-path it cannot be caught by normal OpenVINO exception handling.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/src/plugin/sync_infer_request.cpp:552` — `SyncInferRequest::setup_stream_graph()`
**Validated for repos:** openvino
**Trust boundary:** m_graphs population at CompiledModel construction time vs. stream_id derived from IStreamsExecutor::get_stream_id() at inference time

## Description / Root cause
At line 551, `auto num_graphs = stream_graphs.size();` — if m_graphs is empty, num_graphs == 0. Line 552 then executes `stream_id = stream_id % num_graphs;` — modulo by zero is undefined behaviour in C++ (typically SIGFPE / crash). The only guard is `if (nullptr != m_stream_executor)` (line 549), which is true during normal async inference, so an empty m_graphs vector combined with a live stream executor will trigger the crash on every infer() call.

**Validator analysis:** CWE-369 Divide-By-Zero is accurate for the modulo at sync_infer_request.cpp:552: when m_stream_executor is non-null, stream_id %= num_graphs, and num_graphs==stream_graphs.size(). The constructors (compiled_model.cpp:64-68 and :159-163) build exactly num_streams graphs; apply_performance_hints (execution_config.cpp:406-433) only special-cases ov::streams::AUTO(-1) and exclusive-async — an explicitly user-set NUM_STREAMS=0 is NOT clamped, so m_graphs ends up empty. Then num_graphs==0 → `% 0` is UB/SIGFPE in the hot path, uncatchable. Note line 555 `stream_graphs[stream_id]` is ALSO an out-of-bounds read on an empty vector even on the no-executor path, so the impact is broader than just the modulo. The DoS impact is accurate but the precondition is application self-misconfiguration (num_streams=0), not attacker-controlled model data — that is why the EP boundary is rejected. The proposed fix (OPENVINO_ASSERT(num_graphs>0)) converts the crash to a catchable exception and is acceptable, but it only covers the modulo; the better/sufficient fix is to clamp num_streams to >=1 during finalize (or assert !m_graphs.empty() at the top of setup_stream_graph) so that line 555's empty-vector indexing is also covered.

## Exploit / Proof of Concept
Same precondition as finding 1: configure CompiledModel with num_streams=0, then call create_infer_request() and invoke infer(). infer() calls setup_stream_graph(); m_stream_executor is non-null (set by the async request); stream_graphs is the empty m_graphs vector; num_graphs = 0; stream_id % 0 is executed, causing UB / crash.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 at
//   openvino/src/plugins/intel_gpu/src/plugin/sync_infer_request.cpp:551-555
// Encodes the fix: with NUM_STREAMS=0 the GPU CompiledModel must NOT leave
// m_graphs empty / must reject inference cleanly instead of executing
// `stream_id % 0` (SIGFPE/UB) or indexing an empty vector at line 555.
// Pre-fix: crashes (FPE / ASan container-overflow on stream_graphs[0]).
// Post-fix: the OPENVINO_ASSERT / num_streams clamp turns it into a throw.
//
// NOTE: requires a real GPU device; symbols/macros below are best-effort and
// must be confirmed against intel_gpu/tests/unit/ before use.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

TEST(gpu_sync_infer_request, num_streams_zero_does_not_divide_by_zero) {
    ov::Core core;
    // TODO: replace with a small valid model fixture available in the GPU unit tree
    //       (e.g. ngraph::builder helpers used elsewhere in intel_gpu/tests/unit/).
    std::shared_ptr<ov::Model> model = nullptr; // TODO: build a trivial 1-op model
    ASSERT_NE(model, nullptr) << "TODO: construct test model";

    // Explicitly request zero streams — currently survives execution_config finalize.
    ov::AnyMap cfg = { ov::num_streams(ov::streams::Num(0)) };

    // After the fix this should either clamp to >=1 (so infer succeeds) or throw
    // at compile/infer time; it must NEVER crash with FPE.
    ASSERT_ANY_THROW({
        auto compiled = core.compile_model(model, "GPU", cfg);
        auto req = compiled.create_infer_request();
        req.infer(); // setup_stream_graph(): stream_id % num_graphs with num_graphs==0
    });
}
```
**Build / run:** Build target: ov_gpu_unit_tests. Run: ov_gpu_unit_tests --gtest_filter='gpu_sync_infer_request.num_streams_zero_does_not_divide_by_zero'. Pre-fix expected failure: SIGFPE (integer divide-by-zero) at sync_infer_request.cpp:552, or under ASan a container-overflow at line 555 `stream_graphs[stream_id]` on an empty vector. Post-fix: the assertion/clamp makes the body either throw (ASSERT_ANY_THROW passes) or run without crashing — adjust assertion once the chosen fix (clamp num_streams>=1 vs OPENVINO_ASSERT) is known. TODO: supply a real model fixture and confirm a GPU device is present in CI.

## Suggested fix
Add a guard before the modulo: `if (num_graphs == 0) { OPENVINO_ASSERT(false, "[GPU] No graphs available for inference"); }` — or equivalently reuse the same assertion fix from finding 1 so that m_graphs can never be empty when callers are reached. At minimum add `OPENVINO_ASSERT(num_graphs > 0, "[GPU] No graphs loaded");` immediately after line 551.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #203.
