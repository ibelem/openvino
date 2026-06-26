// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-681/CWE-190 in intel_cpu Gather (gather.h:92, gather.cpp:164/173/427).
// Pre-fix: axisDim is a signed 32-bit int; a data-axis dim > INT_MAX truncates to a
// negative value and the unchecked axisDim*afterAxisSizeInBytes multiply wraps a uint64_t
// stride that the JIT kernel later uses for pointer arithmetic (OOB / corrupt stride).
// Post-fix: axisDim is widened (int64_t/Dim) and the chained shape-product multiplies are
// guarded, so building/compiling a Gather with such a shape must be rejected rather than
// silently producing a wrapped stride.
//
// SKELETON: a self-contained, compilable repro needs either a crafted IR/ONNX with a
// 2^31+1 Gather data-axis dimension or direct construction of the intel_cpu Gather node
// with a mocked >INT_MAX static data shape. Exact ov_cpu_unit_tests fixture/symbol names
// for driving a single node must be confirmed against the surrounding test tree.
#include <gtest/gtest.h>
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(GatherCpuShapeOverflow, AxisDimAboveIntMaxIsRejected) {
    // TODO: confirm the correct ov_cpu_unit_tests harness/helper for compiling a single
    // CPU node (see intel_cpu/tests/unit and the Subgraph/SingleLayer test utilities).
    constexpr int64_t kHugeAxis = (int64_t)std::numeric_limits<int>::max() + 2; // > INT_MAX

    auto data    = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{kHugeAxis, 1});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model   = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data, indices});

    Core core;
    // Pre-fix: compilation builds the node and computes a truncated/wrapped stride in
    // Gather::initSupportedPrimitiveDescriptors (gather.cpp:173) with no diagnostic.
    // Post-fix: the narrowing/overflow guard must reject the oversized axis dimension.
    // TODO: if compile_model does not surface the guard, drive Gather::prepareParams
    // directly and assert CPU_NODE_THROW fires.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
