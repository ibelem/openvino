// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-197/CWE-194 -> CWE-125/CWE-787 in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164 (and :418)
//   openvino/src/plugins/intel_cpu/src/nodes/gather.h:92  (`int axisDim`)
//
// Pre-fix: a Gather whose DATA input declares an axis dimension > INT_MAX is
// silently truncated to a negative `int axisDim`; lines 172-173 sign-extend it
// into a ~UINT64_MAX stride, leading to OOB access in execReference (line 950/954).
// Post-fix (widen to int64_t + bounds/overflow throw): compile_model must reject
// the oversized dimension with an ov::Exception BEFORE any inference/allocation.
//
// NOTE: This builds the graph with the DATA tensor as a *Parameter* (not a
// Constant) so no multi-GB buffer is allocated at compile time; the overflow
// is detected purely from the declared static shape during
// Gather::initSupportedPrimitiveDescriptors().
//
// TODO: confirm the exact intel_cpu unit-test target/harness and includes by
// reading src/plugins/intel_cpu/tests/unit/ ; symbol names below are best-effort.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

TEST(GatherCpuAxisDimTruncation, RejectsOversizedAxisDimension) {
    // axis dimension > INT_MAX -> would truncate to a negative `int axisDim`
    constexpr uint64_t kOversized = 0x80000001ULL;  // 2^31 + 1

    auto data = std::make_shared<ov::op::v0::Parameter>(
        ov::element::u8,
        ov::Shape{static_cast<size_t>(kOversized), 1});
    auto indices = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::Shape{1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto gather = std::make_shared<ov::op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{gather->output(0)},
        ov::ParameterVector{data, indices});

    ov::Core core;
    // Pre-fix: compiles fine (truncation is silent); first inference would OOB.
    // Post-fix: the bounds/overflow check throws here.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
