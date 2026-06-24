// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143 (Range::getWorkAmount, i32 path)
// Pre-fix: a Range op with int32 step==0 reaches div_up(|span|,0) -> integer SIGFPE
//          (assert(b) in general_utils.h:32 is stripped under NDEBUG).
// Post-fix: getWorkAmount must reject step==0 (OPENVINO_THROW), so inference throws
//          ov::Exception instead of crashing the worker thread.
//
// TODO(harness): place under openvino/src/plugins/intel_cpu/tests/unit/ (target ov_cpu_unit_tests)
//   or as a single-layer test under tests/functional (target ov_cpu_func_tests).
// TODO(symbols): confirm exact include paths / fixture base class by reading the
//   neighbouring CPU Range tests before use; symbol names below are best-effort.

#include <gtest/gtest.h>
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// Builds a tiny model: Range(start=0, limit=4, step=<delta>) with i32 scalars.
static std::shared_ptr<Model> make_range_model() {
    auto start = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto limit = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto delta = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto range = std::make_shared<opset8::Range>(start, limit, delta, element::i32);
    return std::make_shared<Model>(OutputVector{range},
                                   ParameterVector{start, limit, delta});
}

TEST(CpuRangeNode, ZeroIntStepDoesNotDivideByZero) {
    Core core;
    auto model = make_range_model();
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    int32_t start = 0, limit = 4, step = 0;  // step == 0 triggers div_up(|span|,0)
    Tensor tStart(element::i32, Shape{}, &start);
    Tensor tLimit(element::i32, Shape{}, &limit);
    Tensor tStep (element::i32, Shape{}, &step);
    req.set_input_tensor(0, tStart);
    req.set_input_tensor(1, tLimit);
    req.set_input_tensor(2, tStep);

    // Pre-fix: SIGFPE crash inside Range::getWorkAmount -> div_up.
    // Post-fix: getWorkAmount throws on step==0, surfaced as ov::Exception.
    EXPECT_THROW(req.infer(), ov::Exception);
}
