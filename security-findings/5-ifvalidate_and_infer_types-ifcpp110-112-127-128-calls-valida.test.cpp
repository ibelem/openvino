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
