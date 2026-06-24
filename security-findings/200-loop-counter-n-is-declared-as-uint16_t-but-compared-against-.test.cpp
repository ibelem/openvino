// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for compiled_model.cpp:65/:160 (uint16_t loop counter vs int32_t
// get_num_streams()). Pre-fix: ov::num_streams > 65535 wraps the uint16_t counter and
// the CompiledModel ctor loops forever (hang / OOM). Post-fix: either the options.inl
// validator rejects the out-of-range value (throws) or the int32_t counter terminates.
// This test asserts the call returns/throws rather than hanging.
//
// Target: ov_gpu_unit_tests (intel_gpu/tests/unit). Requires a GPU device.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

TEST(gpu_compiled_model, num_streams_overflow_is_bounded) {
    ov::Core core;
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);
    auto r = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p});

    // Pre-fix this hangs (uint16_t counter wraps at 65535); post-fix the validator
    // rejects the value at the config boundary -> throws ov::Exception.
    EXPECT_THROW(core.compile_model(model, "GPU", {ov::num_streams(100000)}), ov::Exception);
}
