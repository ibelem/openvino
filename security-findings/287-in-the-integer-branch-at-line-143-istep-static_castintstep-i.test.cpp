// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for CWE-369 divide-by-zero at
// openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143 (Range::getWorkAmount).
// Pre-fix: an i32 Range node with runtime delta==0 reaches div_up(abs_span, 0)
// -> integer SIGFPE / crash. Post-fix (step==0 guarded with a throw): the
// compiled model's infer request rejects step==0 instead of crashing.
//
// Harness: ov_cpu_unit_tests / CPU functional tests (intel_cpu/tests).
// SKELETON: exact builder helper names and the CPU-device infer entry point
// must be confirmed against the existing intel_cpu Range tests before use.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_Range_div_by_zero, i32_runtime_delta_zero_must_not_crash) {
    // Build: Range(start, stop, step) all i32 scalars as runtime Parameters
    // so step is NOT const-folded and the CPU node executes getWorkAmount.
    auto start = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto stop  = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto step  = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto model = std::make_shared<Model>(OutputVector{range},
                                         ParameterVector{start, stop, step});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_start(element::i32, Shape{}); t_start.data<int32_t>()[0] = 0;
    Tensor t_stop (element::i32, Shape{}); t_stop.data<int32_t>()[0]  = 10;
    Tensor t_step (element::i32, Shape{}); t_step.data<int32_t>()[0]  = 0; // delta == 0
    req.set_input_tensor(0, t_start);
    req.set_input_tensor(1, t_stop);
    req.set_input_tensor(2, t_step);

    // TODO: confirm the post-fix behaviour: getWorkAmount should throw on step==0.
    // Pre-fix this line SIGFPEs (test process crashes); post-fix it must throw.
    EXPECT_ANY_THROW(req.infer());
}
