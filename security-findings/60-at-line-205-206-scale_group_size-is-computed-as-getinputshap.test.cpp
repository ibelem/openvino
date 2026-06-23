// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205-206 (GATHER_SCALE)
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:217-218 (GATHER_ZP)
// where scale_group_size / zp_group_size are computed as
//   GATHER_DATA.getElementsCount() / GATHER_SCALE(or ZP).getElementsCount()
// with no zero check. A GatherCompressed node whose scale/zp input has a static
// zero-dim shape (e.g. [1024, 0]) makes the denominator 0 -> SIGFPE during
// Gather::initSupportedPrimitiveDescriptors (i.e. at compile time).
//
// This test builds a GatherCompressed (ov::op::internal::GatherCompressed) graph
// with a zero-dim scale constant and compiles it on CPU. PRE-FIX: the process
// dies with SIGFPE (integer division by zero) inside the CPU Gather node.
// POST-FIX: Gather::initSupportedPrimitiveDescriptors throws via CPU_NODE_ASSERT,
// surfaced as an ov::Exception by core.compile_model, which we assert below.
//
// Harness: ov_cpu_unit_tests (the intel_cpu component test target).
// NOTE: SKELETON — exact GatherCompressed factory signature and the test helper
// includes must be confirmed against intel_cpu/tests/unit before use.

#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

// TODO: confirm the correct internal op header path for GatherCompressed in this
//       tree (search intel_cpu transformations for ov::op::internal::GatherCompressed).
// #include "transformations/op_conversions/.../gather_compressed.hpp"  // TODO: real path

using namespace ov;

TEST(CpuGatherCompressed, ZeroDimScaleShapeIsRejectedNotSIGFPE) {
    // data: u8 [1024, 16]
    auto data = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{1024, 16});
    // indices: i32 [4]
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    // axis const = 0
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    // scale const with a ZERO dimension -> getElementsCount() == 0 (the divisor)
    auto scale = op::v0::Constant::create(element::f32, Shape{1024, 0}, std::vector<float>{});

    // TODO: construct ov::op::internal::GatherCompressed with the exact ctor
    //       (data, indices, axis, batch_dims, scale[, zp]). Confirm arg order.
    // auto gc = std::make_shared<op::internal::GatherCompressed>(data, indices, axis, 0, scale);
    // auto model = std::make_shared<Model>(OutputVector{gc},
    //                                      ParameterVector{data, indices});

    // ov::Core core;
    // EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
    GTEST_SKIP() << "TODO: wire up GatherCompressed ctor + model, then enable EXPECT_THROW";
}
