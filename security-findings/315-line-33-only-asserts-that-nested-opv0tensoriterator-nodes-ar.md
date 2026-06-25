# Security finding #315: Line 33 only asserts that nested `op::v0::TensorIterator` nodes are…

**Summary:** Line 33 only asserts that nested `op::v0::TensorIterator` nodes are…

**CWE IDs:** CWE-674: Uncontrolled Recursion
**Severity / Impact:** Stack overflow / process crash (DoS) triggered purely by model topology. Any application loading an attacker-supplied ONNX model that contains a v5::Loop nested inside a TensorIterator body (which itself may contain further nesting) will exhaust the native call stack during shape inference, crashing the process.
**Affected location:** `targets/openvino/src/core/src/op/tensor_iterator.cpp:33` — `op::v0::TensorIterator::revalidate_and_infer_types_for_body_ops()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file → TensorIterator body graph construction → shape inference

## Description / Root cause
Line 33 only asserts that nested `op::v0::TensorIterator` nodes are absent, but places NO guard against `op::v5::Loop` nodes in the body. At line 45, `node->revalidate_and_infer_types()` is called unconditionally for any other node type including `v5::Loop`. `Loop::validate_and_infer_types()` (loop.cpp:206) calls `m_bodies[0]->validate_nodes_and_infer_types()`, which iterates over body nodes and calls `validate_and_infer_types()` on each — including any nested TensorIterator, which calls `revalidate_and_infer_types_for_body_ops()` again. An arbitrarily deep Loop→TI→Loop→TI… nesting drives unbounded native-stack recursion with no depth counter anywhere on the path.

**Validator analysis:** The flaw is real in OpenVINO core. revalidate_and_infer_types_for_body_ops (tensor_iterator.cpp:23-52) uses an explicit work-stack to avoid recursion WITHIN a single body, but its only topology guard (line 33) blocks a directly-nested v0::TensorIterator and nothing else. A v5::Loop node satisfies the assert, so line 45 invokes node->revalidate_and_infer_types(), which is Loop::validate_and_infer_types; that method at loop.cpp:206 calls m_bodies[0]->validate_nodes_and_infer_types(), iterating the inner graph and calling validate_and_infer_types() on each node — a nested TensorIterator there re-enters revalidate_and_infer_types_for_body_ops. This Loop→TI→Loop→TI chain is genuine native-stack recursion with no depth counter, so CWE-674 (Uncontrolled Recursion) and the stack-overflow/DoS impact are accurate for model-topology-driven input. The reachability claim is sound for OpenVINO's own ONNX frontend / read_model path (validate_and_infer_types runs at model build). The proposed fix is PARTLY WRONG: the suggested `OPENVINO_ASSERT(as_type_ptr<v5::Loop>(node)==nullptr, "No nested Loop")` would reject legitimate, valid models that nest a Loop inside a TI body — that is a correctness regression, not a fix. The correct/sufficient fix is the depth-counter alternative: a thread_local recursion-depth guard shared across SubGraphOp::validate_and_infer_types (TensorIterator and Loop) that increments on entry, decrements on exit, and throws ov::Exception past a safe bound (e.g. 64). It should live in the common util::SubGraphOp path so Loop, TensorIterator and Scan are all covered, rather than only patching TensorIterator.

## Exploit / Proof of Concept
Craft an ONNX model with: TensorIterator (body contains v5::Loop (body contains TensorIterator (body contains v5::Loop …))). Depth of ~500–1000 nesting levels is sufficient on most platforms. When OpenVINO parses and calls `TensorIterator::validate_and_infer_types()`, it calls `revalidate_and_infer_types_for_body_ops()` (ti.cpp:120), which at line 45 calls `Loop::validate_and_infer_types()`, which at loop.cpp:206 calls `validate_nodes_and_infer_types()` on the inner body, which calls the inner TI's `validate_and_infer_types()`, recursing back into `revalidate_and_infer_types_for_body_ops()`. No depth guard interrupts this chain.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-674 in tensor_iterator.cpp:33-45 / loop.cpp:206.
// Pre-fix: deeply alternating TensorIterator<->Loop nesting drives unbounded
//   native recursion through revalidate_and_infer_types_for_body_ops ->
//   Loop::validate_and_infer_types -> validate_nodes_and_infer_types -> inner TI,
//   crashing with a stack overflow (no depth guard anywhere on the path).
// Post-fix: a recursion-depth guard must throw ov::Exception instead of recursing.
//
// Harness: ov_core_unit_tests (gtest). Target component: openvino/src/core.
// TODO: confirm exact target/file location by reading the nearest existing
//       core op shape-inference tests (e.g. src/core/tests/type_prop/*).
// TODO: verify helper symbol names (make_shared op builders, set_function,
//       set_invariant_input/get_iter_value, etc.) against the real headers —
//       these are best-effort and likely need adjustment.
#include <gtest/gtest.h>

#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

namespace {
// Build a trivial identity body: Parameter -> Result, wrapped alternately in
// Loop / TensorIterator to a given nesting depth.
std::shared_ptr<Node> build_nested(size_t depth) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    std::shared_ptr<Node> inner = p;
    for (size_t i = 0; i < depth; ++i) {
        auto bp = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
        auto br = std::make_shared<op::v0::Result>(bp);
        auto body = std::make_shared<Model>(ResultVector{br}, ParameterVector{bp});
        if (i % 2 == 0) {
            auto ti = std::make_shared<op::v0::TensorIterator>();
            ti->set_function(body);
            ti->set_invariant_input(bp, inner);
            inner = ti->get_iter_value(br, -1).get_node_shared_ptr();
        } else {
            auto loop = std::make_shared<op::v5::Loop>(/* trip_count */ inner,
                                                       /* cond */ inner);
            loop->set_function(body);
            // TODO: wire loop input/output descriptions correctly for v5::Loop.
            inner = loop;
        }
    }
    return inner;
}
}  // namespace

TEST(type_prop, tensor_iterator_loop_nesting_depth_guard) {
    // ~256 alternating levels: well past any sane threshold, far below what a
    // native stack survives, so pre-fix this recurses to a crash.
    auto top = build_nested(256);
    // Post-fix the depth guard must convert this into a bounded ov::Exception
    // rather than overflowing the stack.
    EXPECT_THROW(top->validate_and_infer_types(), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests. Run: ./ov_core_unit_tests --gtest_filter=type_prop.tensor_iterator_loop_nesting_depth_guard. Pre-fix expected failure: SIGSEGV / AddressSanitizer 'stack-overflow' inside op::v0::TensorIterator::revalidate_and_infer_types_for_body_ops <-> op::v5::Loop::validate_and_infer_types recursion. Post-fix expected: test passes (ov::Exception thrown by the recursion-depth guard). NOTE: skeleton — the v5::Loop input/output-description wiring and op-builder helper names are placeholders and must be corrected against the real core headers before it will compile.

## Suggested fix
Add a thread-local or parameter-passing depth counter. Before calling `node->revalidate_and_infer_types()` at line 45, assert (or check and throw) that no `op::v5::Loop` or any other `SubGraphOp` subclass is present, mirroring the existing TensorIterator guard: `OPENVINO_ASSERT(ov::as_type_ptr<op::v5::Loop>(node) == nullptr, "No nested Loop");`. Better yet, introduce a `static thread_local int depth` counter in both `revalidate_and_infer_types_for_body_ops` and `Loop::validate_and_infer_types`, incrementing on entry and decrementing on exit, and throwing `ov::Exception` when depth exceeds a safe threshold (e.g. 32).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #315.
