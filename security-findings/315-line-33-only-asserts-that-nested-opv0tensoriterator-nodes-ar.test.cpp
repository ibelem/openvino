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
