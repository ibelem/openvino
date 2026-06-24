// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read at
//   openvino/src/plugins/intel_cpu/src/nodes/reshape.cpp:82
//   `lastSecondInputValues.resize(mem.getStaticDims()[0], 0);`
// where getStaticDims() (cpu_shape.h:127-130) returns an EMPTY VectorDims for a
// rank-0 (scalar) static second input, so operator[](0) reads out of bounds.
//
// This encodes the fix: a Reshape node whose second (shape) input is a static
// rank-0 scalar must be REJECTED (CPU_NODE_THROW) instead of dereferencing
// element [0] of an empty dims vector. Pre-fix this aborts under ASan with a
// container-overflow / heap-buffer-overflow in needShapeInfer(); post-fix the
// empty-dims guard throws ov::Exception during node construction or shape infer.
//
// NOTE (why this is a SKELETON): exercising Node::needShapeInfer() in isolation
// requires a fully built ov::intel_cpu::Graph with allocated parent edges and
// backing Memory for input 1, which the read-only tree does not expose via a
// simple unit entry point. The exact construction helpers must be copied from
// an existing ov_cpu_unit_tests fixture. TODOs below name what is missing.

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

using namespace ov;

TEST(ReshapeNodeOOB, scalar_shape_input_is_rejected_not_oob) {
    // Build a v1::Reshape with a DYNAMIC data input (so the CPU node is dynamic
    // and needShapeInfer() is on the hot path) and a STATIC RANK-0 scalar shape
    // pattern (PartialShape{} -> is_dynamic()==false, rank()==0). This is the
    // exact shape that slips past the constructor guard at reshape.cpp:59.
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto pattern = op::v0::Constant::create(element::i64, ov::Shape{} /* rank-0 scalar */, {1});
    const auto reshape = std::make_shared<op::v1::Reshape>(data, pattern, /*special_zero=*/true);

    // TODO: build the intel_cpu Graph node from `reshape` using the same helper
    //       used in intel_cpu/tests/unit (e.g. cpuNodeFromOp / GraphContext) and
    //       allocate a backing Memory for parent edge 1 with a rank-0 descriptor.
    // TODO: replace the line below with the node construction + needShapeInfer()
    //       invocation once the helper symbol is confirmed by reading the
    //       surrounding intel_cpu unit test fixtures.
    //
    // Expected post-fix behaviour: rank-0 second input is rejected.
    //   EXPECT_THROW(makeAndInferCpuReshape(reshape), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu Graph/Node fixture to call needShapeInfer()";
}
