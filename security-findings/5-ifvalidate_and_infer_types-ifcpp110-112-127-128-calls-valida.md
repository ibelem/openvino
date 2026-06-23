# Security finding #5: If::validate_and_infer_types (if.cpp:110, 112, 127, 128) calls vali…

**Summary:** If::validate_and_infer_types (if.cpp:110, 112, 127, 128) calls vali…

**CWE IDs:** CWE-674: Uncontrolled Recursion
**Severity / Impact:** Stack exhaustion / process crash (DoS). Any caller invoking Core::read_model() or any path that triggers shape inference on a deeply nested If-inside-If model will cause an unhandled stack overflow, crashing the host process. Affects all consumers of the OpenVINO runtime when loading untrusted model files.
**Affected location:** `targets/openvino/src/core/src/op/if.cpp:78` — `ov::op::v8::If::validate_and_infer_types()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted model file → IR/ONNX deserializer → If/Loop node construction → validate_and_infer_types called with no nesting depth limit

## Description / Root cause
If::validate_and_infer_types (if.cpp:110, 112, 127, 128) calls validate_and_infer_type_body(), which at multi_subgraph_base.cpp:164 unconditionally calls body->validate_nodes_and_infer_types(). That function at model.cpp:246 calls node->revalidate_and_infer_types() for every node in the subgraph body. If any body node is itself an If (or Loop), its own validate_and_infer_types fires, re-entering the same chain with no depth counter, visited-set, or stack-limit guard at any level.

**Validator analysis:** The recursion is real and unguarded: If::validate_and_infer_types (if.cpp:78) calls validate_and_infer_type_body (multi_subgraph_base.cpp:151), which at line 164 unconditionally calls body->validate_nodes_and_infer_types(); that re-validates every body node, so an inner If re-enters the same chain one stack frame deeper per nesting level. There is no depth counter, visited-set, or stack-limit check in any of these functions, confirming CWE-674. The impact (native stack exhaustion -> uncatchable crash/DoS during read_model/shape inference of an attacker-crafted IR or ONNX with N nested If/Loop bodies) is accurate; note a surrounding try/catch in read_model would NOT mitigate a stack overflow, so a preventive depth check is genuinely required. validated for openvino (the bug exists and is reachable from Core::read_model). For openvinoEp I reject: the vulnerable code is not in plugin_impl, and I cannot confirm that a fully nested If-in-If structure is forwarded as a single OpenVINO model through the EP (ORT control-flow delegation makes this unproven), so per a skeptical stance reachability from that boundary is not established (not 'na' because propagation is plausible, just unconfirmed). The proposed fix (thread-local depth counter incremented in validate_and_infer_type_body with throw past a limit) is directionally correct and sufficient to prevent the overflow, but it must use an RAII guard so the counter decrements even when NODE_VALIDATION_CHECK throws (a bare --depth before throw misses exceptions raised by body validation itself); a limit of 512 is reasonable. An equally valid alternative is a nesting-depth limit enforced in the IR/ONNX frontend when building If/Loop bodies.

## Exploit / Proof of Concept
Craft an IR or ONNX model containing N nested If nodes (body of If_1 contains If_2, body of If_2 contains If_3, …). When deserialized and submitted to shape inference, each If::validate_and_infer_types recurses one level deeper per nest. With N ≈ 1000–3000 (well within what an IR XML can encode), the native call stack (typically 1–8 MB) is exhausted before any exception is raised, crashing the process. No upstream check in the reader or in any of the three functions rejects excessive nesting depth.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-674 uncontrolled recursion in
//   targets/openvino/src/core/src/op/if.cpp:78 (validate_and_infer_types)
//   -> targets/openvino/src/core/src/op/util/multi_subgraph_base.cpp:164
//      (body->validate_nodes_and_infer_types() with no depth guard)
// PRE-FIX: building a deeply nested If-inside-If model and triggering shape
//   inference recurses one frame per nesting level and overflows the native
//   stack (ASan: stack-overflow / SEGV) before any exception is raised.
// POST-FIX: a depth guard in validate_and_infer_type_body throws ov::Exception
//   once MAX_SUBGRAPH_DEPTH is exceeded, so the model is rejected cleanly.
//
// Harness: ov_core_unit_tests (gtest). Place alongside src/core/tests/type_prop/if.cpp.
// NOTE: marked skeleton — the recursive If builder below uses the public op API
//   but exact helper/description symbols should be confirmed against
//   src/core/tests/type_prop/if.cpp before relying on it.

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

namespace {
// Build an If whose then/else bodies each contain `depth-1` further nested Ifs.
// At depth 0 the body is a trivial pass-through identity model.
std::shared_ptr<Model> make_nested_if_body(size_t depth) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    if (depth == 0) {
        auto r = std::make_shared<op::v0::Result>(p);
        return std::make_shared<Model>(ResultVector{r}, ParameterVector{p});
    }
    auto cond = op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto if_op = std::make_shared<op::v8::If>(cond);
    auto then_body = make_nested_if_body(depth - 1);
    auto else_body = make_nested_if_body(depth - 1);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    // TODO: confirm exact set_input/set_output description API names from
    //       src/core/tests/type_prop/if.cpp (set_input/set_output overloads).
    if_op->set_input(p, then_body->get_parameters()[0], else_body->get_parameters()[0]);
    auto out = if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
    auto r = std::make_shared<op::v0::Result>(out);
    return std::make_shared<Model>(ResultVector{r}, ParameterVector{p});
}
} // namespace

TEST(type_prop, if_excessive_subgraph_nesting_is_rejected) {
    // Choose a depth well below what is needed to overflow an 8MB stack at
    // runtime but above the intended MAX_SUBGRAPH_DEPTH guard (e.g. 512).
    constexpr size_t kDepth = 4000;
    // Pre-fix this construction/validation overflows the stack (ASan crash).
    // Post-fix it must throw cleanly instead of crashing.
    EXPECT_THROW(make_nested_if_body(kDepth), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_core_unit_tests --gtest_filter='type_prop.if_excessive_subgraph_nesting_is_rejected'. Pre-fix expected: AddressSanitizer 'stack-overflow' (or SEGV from recursive validate_and_infer_type_body) and the test aborts. Post-fix expected: the guard throws ov::Exception ('Subgraph nesting depth exceeded') and the EXPECT_THROW passes. TODO before use: verify If set_input/set_output description helper signatures against src/core/tests/type_prop/if.cpp.

## Suggested fix
Introduce a thread-local (or context-passed) recursion depth counter in validate_and_infer_type_body (multi_subgraph_base.cpp:151). Increment it on entry and decrement on exit; throw an ov::Exception if it exceeds a compile-time limit (e.g., 512). Alternatively, add a max-nesting depth check in the IR/ONNX frontend when constructing If/Loop subgraph nodes. A minimal change: add 'static thread_local int depth = 0; if (++depth > MAX_SUBGRAPH_DEPTH) { --depth; OPENVINO_THROW("Subgraph nesting depth exceeded"); }' at the top of validate_and_infer_type_body, with a decrement-on-return guard.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #5.
