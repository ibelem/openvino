# Security finding #12: Line 206 calls `m_bodies[0]->validate_nodes_and_infer_types()`, whi…

**Summary:** Line 206 calls `m_bodies[0]->validate_nodes_and_infer_types()`, whi…

**CWE IDs:** CWE-674: Uncontrolled Recursion
**Severity / Impact:** Stack exhaustion / crash (DoS). An attacker who can supply a crafted model (ONNX or OpenVINO IR) containing N levels of nested Loop ops can cause the validation stack to grow to depth ~N. With default OS stack sizes (~1–8 MB) and typical frame sizes, a few hundred to a few thousand nested Loops is sufficient to overflow the stack and terminate the host process.
**Affected location:** `targets/openvino/src/core/src/op/loop.cpp:206` — `Loop::validate_and_infer_types()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted model file → ONNX/IR deserializer → Loop node construction and graph validation

## Description / Root cause
Line 206 calls `m_bodies[0]->validate_nodes_and_infer_types()`, which in `Model::validate_nodes_and_infer_types` (model.cpp:246) iterates all body nodes and calls `node->revalidate_and_infer_types()` on each. If a body node is itself a `Loop`, this re-enters `Loop::validate_and_infer_types()`, which calls the inner body's `validate_nodes_and_infer_types()` again. Line 267 repeats the same call inside the back-edge reconciliation loop. There is no recursion depth counter, depth cap, or visited-set anywhere in this chain.

**Validator analysis:** The recursion is real and confirmed: Loop::validate_and_infer_types (loop.cpp:206, and again at 267 inside the back-edge reconciliation loop) calls the body Model's validate_nodes_and_infer_types(); Model::validate_nodes_and_infer_types (model.cpp:245-246) iterates get_ordered_ops() and calls revalidate_and_infer_types() on each node, which for a nested Loop re-enters Loop::validate_and_infer_types. No depth cap, visited-set, or worklist exists, so recursion depth tracks the model's Loop-nesting depth 1:1. OPENVINO_ASSERT/NODE_VALIDATION_CHECK cannot mitigate this — a native stack overflow is not a catchable C++ exception, so the surrounding validation throws do NOT save it. CWE-674 (Uncontrolled Recursion) and the stack-exhaustion/DoS impact are accurate. Reachability is genuine: nested Loop subgraphs are valid ONNX/IR, and read_model/Model-construction/compile_model all trigger validate_and_infer_types; the EP is a pass-through that hands attacker-controlled ONNX to OpenVINO. The proposed fix (RAII thread-local depth guard that throws past a hard cap) is correct and sufficient to convert the uncatchable crash into a catchable validation error for THIS path; a cap of 64 is safe (real models never nest Loops that deeply), though I'd recommend a slightly higher cap (e.g. 256) and note that other recursive subgraph paths (ONNX frontend subgraph parsing, clone_model) share the same unbounded-depth class and should ideally route through the same shared depth guard. The iterative-worklist alternative is more invasive but would also fix it.

## Exploit / Proof of Concept
Craft a model whose outermost Loop body contains a Loop, whose body in turn contains another Loop, and so on — N levels deep, each with a single trivial back-edge. When any API that triggers graph validation (e.g., `ov::Core::compile_model`, `ov::Model` construction, or `read_model`) is called with this model, `Loop::validate_and_infer_types` recurses N times through `Model::validate_nodes_and_infer_types → revalidate_and_infer_types → Loop::validate_and_infer_types`. N ≈ 2000–5000 exhausts the default thread stack, producing a stack overflow crash.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-674 in ov::op::v5::Loop::validate_and_infer_types
//   targets/openvino/src/core/src/op/loop.cpp:206 and :267
//   (m_bodies[0]->validate_nodes_and_infer_types() re-enters Loop validation
//    for every nested Loop with no recursion depth cap -> stack overflow).
//
// What this encodes:
//   - Build N deeply-nested Loop ops (each body contains a Loop).
//   - PRE-FIX: validation recurses N deep and overflows the stack (crash;
//     ASan/native SIGSEGV, NOT catchable by EXPECT_THROW).
//   - POST-FIX: the depth guard throws ov::Exception before exhausting the
//     stack, so the construction is rejected cleanly.
//
// Harness: ov_core_tests (gtest). Place in the nearest core op test dir,
// e.g. targets/openvino/src/core/tests/type_prop/loop.cpp
// TODO: confirm exact test file + target name by reading the surrounding
//       core tests tree before adding.

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

// TODO: factor a helper that returns a minimal Loop whose body is `inner`
//       (a trivial body when inner == nullptr). Exact SpecialBodyPorts /
//       trip-count / exec-condition wiring must be copied from an existing
//       passing Loop type_prop test — do not guess the port-map indices.
static std::shared_ptr<op::v5::Loop> make_nested_loop(size_t depth) {
    // TODO: implement nesting:
    //   for i in [0, depth): wrap the current body Model in a new Loop whose
    //   single body node is that Loop, with one trivial merged/back-edge input.
    //   Each level must add a back-edge so loop.cpp:267 is also exercised.
    (void)depth;
    return nullptr; // TODO
}

TEST(type_prop, loop_deeply_nested_recursion_is_bounded) {
    constexpr size_t kDepth = 4000;  // exceeds default thread stack pre-fix
    auto outer = make_nested_loop(kDepth);
    // POST-FIX expectation: a hard recursion-depth cap rejects the model
    // instead of overflowing the stack.
    EXPECT_THROW(outer->validate_and_infer_types(), ov::Exception);
}
```
**Build / run:** Build target: ov_core_tests. Run: ov_core_tests --gtest_filter='type_prop.loop_deeply_nested_recursion_is_bounded'. Pre-fix: native stack overflow / SIGSEGV (under ASan: 'stack-overflow' / 'stack-buffer-overflow' or AddressSanitizer:DEADLYSIGNAL) — test process crashes rather than asserting. Post-fix: the depth-guard throws ov::Exception and the test passes. NOTE: the make_nested_loop helper is a TODO — copy exact Loop body/port-map construction from an existing passing type_prop Loop test before use; a crafted nested-Loop .onnx fixture run through ov_onnx_frontend_tests' convert_model(...) wrapped in EXPECT_THROW is an equally valid alternative.

## Suggested fix
Introduce a thread-local or context-carried recursion depth counter (e.g., `static thread_local int g_loop_validate_depth = 0;`). Increment it on entry to `Loop::validate_and_infer_types`, assert/throw if it exceeds a hard limit (e.g., 64), and decrement on exit (RAII guard). Alternatively, refactor back-edge reconciliation to be iterative using an explicit worklist stack rather than relying on implicit call-stack recursion through the model graph.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #12.
