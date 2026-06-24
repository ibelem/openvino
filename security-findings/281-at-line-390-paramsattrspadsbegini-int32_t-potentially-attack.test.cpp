// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195 at openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:390
// (srcODims[i] = padsBegin[i] + srcDims[i] with signed->unsigned wrap on negative padsBegin).
// Pre-fix: a Pad with dynamic pads_begin = -100 and pads_end = +100 on a dim of size 5
// yields a valid positive output dim but srcODims wraps to ~SIZE_MAX, so the bounds guard at
// pad.cpp:509/554 is bypassed and srcIdx at pad.cpp:522/567 reads out of bounds (ASan heap-buffer-overflow).
// Post-fix: paramsInitialization/workPartition must reject pads_begin < -srcDims (OPENVINO_THROW / ov::Exception).
//
// NOTE (skeleton): exact harness target/symbols for intel_cpu single-layer tests are unverified.
// Build under the CPU functional/unit test tree (e.g. ov_cpu_func_tests / ov_cpu_unit_tests).

#include <gtest/gtest.h>
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(intel_cpu_Pad_NegativeCrop, dynamic_pads_begin_oob_is_rejected) {
    // TODO: confirm the canonical CPU single-layer test fixture (see
    // src/plugins/intel_cpu/tests/functional/.../single_layer_tests/pad.cpp) and reuse it instead.
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{5});
    // dynamic (Parameter) pads => shapeHasDataDependency == true, no construction-time validation.
    auto pads_begin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pads_end   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pad_value  = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto pad = std::make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad},
                                         ParameterVector{data, pads_begin, pads_end});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::f32, Shape{5});
    std::fill_n(t_data.data<float>(), 5, 1.0f);
    Tensor t_begin(element::i32, Shape{1});
    Tensor t_end(element::i32, Shape{1});
    t_begin.data<int32_t>()[0] = -100; // crop far beyond dim -> srcODims wraps pre-fix
    t_end.data<int32_t>()[0]   = 100;  // compensate so output dim stays positive (=5), allocation succeeds

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_begin);
    req.set_input_tensor(2, t_end);

    // Pre-fix: ASan reports heap-buffer-overflow inside Pad::PadExecutor::padConstant* (OOB src read).
    // Post-fix: the negative crop exceeding the dim must be rejected with an ov::Exception.
    EXPECT_THROW(req.infer(), ov::Exception);
}
