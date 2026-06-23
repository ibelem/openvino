// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-197 numeric truncation in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164 (and :418)
//   `int axisDim = dataDims[axis];`  // gather.h:92 declares axisDim as int
// Pre-fix: a data axis dimension > INT_MAX truncates to a negative int,
//   then sign-extends into uint64_t byte strides (gather.cpp:172-175/426-429)
//   that are handed to the JIT kernel (execute:491-494). The fix must reject
//   such an axis dim (CPU_NODE_ASSERT) or carry it as size_t end-to-end.
//
// This test exercises Gather::initSupportedPrimitiveDescriptors via the CPU
// node construction path with a data shape whose axis dim exceeds INT32_MAX.
// It asserts construction/desc-init throws (post-fix) instead of silently
// computing corrupted strides.
//
// NOTE: SKELETON — the exact intel_cpu unit-test harness symbols for directly
// instantiating a Gather node (GraphContext, dummy ov::op::v8::Gather with a
// huge PartialShape) must be confirmed against the surrounding test tree under
// src/plugins/intel_cpu/tests/unit/ before this will compile. Allocating a real
// 2^31+1-element tensor is NOT required: the truncation happens at shape-init
// time (initSupportedPrimitiveDescriptors), before any execution/allocation.

#include <gtest/gtest.h>
#include <limits>
#include <memory>

#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: include the intel_cpu node + graph-context headers used by the existing
// unit tests, e.g.:
//   #include "nodes/gather.h"
//   #include "graph_context.h"
// and any helper that builds a Node from an ov::Node (see tests/unit/).

using namespace ov;

TEST(GatherCpuNodeTest, RejectsAxisDimExceedingInt32Range) {
    // Data shape with axis (=0) dimension just past INT32_MAX.
    const size_t hugeDim = static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1;
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{static_cast<int64_t>(hugeDim), 1});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);

    // TODO: replace with the repo's actual node-construction helper, e.g.:
    //   auto ctx  = std::make_shared<intel_cpu::GraphContext>(/* config, ... */);
    //   auto node = std::make_shared<intel_cpu::node::Gather>(gather, ctx);
    //   EXPECT_THROW(node->initSupportedPrimitiveDescriptors(), ov::Exception);
    //
    // Pre-fix: no throw; axisDim truncates to a negative int and feeds
    //   axisAndAfterAxisSizeInBytes/srcAfterBatchSizeInBytes as bogus strides.
    // Post-fix: a CPU_NODE_ASSERT on dataDims[axis] <= INT32_MAX throws here.
    GTEST_SKIP() << "Wire up intel_cpu Gather node construction helper from "
                    "src/plugins/intel_cpu/tests/unit/ before enabling.";
}
