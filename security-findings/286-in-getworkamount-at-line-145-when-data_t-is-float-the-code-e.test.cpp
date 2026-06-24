// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143,145
// (Range::getWorkAmount / rangeKernel) where a zero `step` causes integer
// divide-by-zero (i32) or a UB float->size_t cast of ceil(inf/NaN) (f32),
// leading to OOM/OOB. Pre-fix: SIGFPE (i32) or huge work_amount_dst -> ASan
// heap-buffer-overflow on dst_data[iwork] (range.cpp:164). Post-fix: the node
// either rejects the zero step (throw) or yields a 0-element output safely.
//
// Harness: ov_cpu_unit_tests (intel_gpu/intel_cpu tests live under
// src/plugins/intel_cpu/tests/unit). EXACT target/fixture names are unverified
// — confirm against the existing single-layer/node test tree before use.
// TODO: locate the correct CPU node test fixture; intel_cpu single-node tests
// usually go through ngraph_functions/ov::Model + ov::Core::compile_model on
// the CPU device, NOT a direct Range node ctor (node ctors need a Graph).

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/constant.hpp"

// TODO: verify device availability guard / SKIP if CPU plugin not built.
TEST(CPURangeZeroStep, F32_ZeroStep_NoCrash) {
    using namespace ov;
    // v4 Range with f32 const inputs, delta = 0.0f (the unguarded path).
    auto start = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto stop  = op::v0::Constant::create(element::f32, Shape{}, {10.0f});
    auto step  = op::v0::Constant::create(element::f32, Shape{}, {0.0f}); // <-- trigger
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::f32);
    auto model = std::make_shared<Model>(OutputVector{range}, ParameterVector{});

    Core core;
    // Pre-fix: compile/infer reaches getWorkAmount with step==0 -> UB cast ->
    // huge work_amount_dst -> ASan heap-buffer-overflow at range.cpp:164 (or OOM).
    // Post-fix: either an ov::Exception is thrown (preferred) or a 0-element
    // output is produced without OOB. Accept either safe outcome.
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto compiled = core.compile_model(model, "CPU");
            auto req = compiled.create_infer_request();
            req.infer();
        } catch (const ov::Exception&) { /* acceptable: input rejected */ }
    });
}

TEST(CPURangeZeroStep, I32_ZeroStep_NoDivByZero) {
    using namespace ov;
    auto start = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto stop  = op::v0::Constant::create(element::i32, Shape{}, {10});
    auto step  = op::v0::Constant::create(element::i32, Shape{}, {0}); // div_up(x,0) -> SIGFPE pre-fix
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto model = std::make_shared<Model>(OutputVector{range}, ParameterVector{});
    Core core;
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto compiled = core.compile_model(model, "CPU");
            compiled.create_infer_request().infer();
        } catch (const ov::Exception&) { /* acceptable */ }
    });
}