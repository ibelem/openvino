// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB in Gather::execCompressed8Bit grouped decompression path.
// Encodes the fix for openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:820 — the inner
// loop `for (size_t g = p; g < p + scale_group_size; g++)` must be clamped to
// srcIdx + afterAxisSize so that when afterAxisSize % scale_group_size != 0 the last
// outer iteration does not read srcData past the row nor write pdst past the per-row
// destination stride. Pre-fix: ASan heap-buffer-overflow (read+write) during execute().
// Post-fix: graph runs and produces correctly sized output without OOB.
//
// Harness: ov_cpu_unit_tests (gtest + ASan). Place near intel_cpu/tests/unit node tests.
// NOTE (skeleton): the internal op ov::op::internal::GatherCompressed is normally produced
// by OpenVINO's weight-decompression fusion, not constructed directly in most unit tests.
// The exact builder helpers / fixture base must be confirmed against the existing
// intel_cpu/tests/unit tree before this compiles.

#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// TODO: confirm the correct fixture/base class and compile target name in
//       src/plugins/intel_cpu/tests/unit/ (e.g. an existing GatherCompressed node test).
TEST(GatherCompressed8Bit_OOB, NonDivisibleScaleGroupSize_NoHeapOverflow) {
    // data: int8 [axisDim=2, afterAxisSize=7]; axis=0; indices select a row.
    // scale shape [2,3] -> scale_group_size = (2*7)/(2*3) = 2 (integer division),
    // afterAxisSize=7, 7 % 2 == 1 -> triggers the unclamped inner loop overrun pre-fix.
    const Shape data_shape{2, 7};
    const Shape scale_shape{2, 3};

    // TODO: replace constant payloads with the precise dtype/precision the CPU plugin
    //       routes through execCompressed8Bit (i8 data, f32 scale, no zero-point => 4 inputs,
    //       non-scalar scale so cond3 is false and the grouped cond2 path is taken).
    auto data  = op::v0::Constant::create(element::i8,  data_shape,  std::vector<int8_t>(2 * 7, 1));
    auto idx   = op::v0::Constant::create(element::i32, Shape{1},    std::vector<int32_t>{0});
    auto axis  = op::v0::Constant::create(element::i32, Shape{},     std::vector<int32_t>{0});
    auto scale = op::v0::Constant::create(element::f32, scale_shape, std::vector<float>(2 * 3, 1.0f));

    auto gc = std::make_shared<op::internal::GatherCompressed>(
        data, idx, axis, /*batch_dims=*/0, scale);

    auto model = std::make_shared<Model>(OutputVector{gc}, ParameterVector{});

    Core core;
    // Pre-fix: ASan reports heap-buffer-overflow inside Gather::execCompressed8Bit.
    // Post-fix: compilation + inference succeed with no OOB.
    EXPECT_NO_THROW({
        auto compiled = core.compile_model(model, "CPU");
        auto req = compiled.create_infer_request();
        req.infer();
    });
}
