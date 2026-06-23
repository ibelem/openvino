// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190 integer overflow in
//   openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408-411
//   (Pad::PadExecutor::innerParamsInitialization)
// Untrusted pad values enter at pad.cpp:282-287 (dynamic case) with no sign check.
// With CONSTANT mode, padsBegin=-100 and padsEnd=+200 on a source dim of 3:
//   inferred output dim = padded(3, +100) = 103   (allocatable -> no bad_alloc),
//   innerCopySize = (3 + min(-100,0) + min(200,0)) * shift = (3-100)*shift -> size_t wraparound.
// Pre-fix: cpu_memcpy at pad.cpp:528/574 copies ~2^64 bytes -> ASan heap-buffer-overflow.
// Post-fix: innerParamsInitialization computes copyLen in int64_t and OPENVINO_THROW on copyLen<0,
//           so inference is rejected with ov::Exception instead of corrupting the heap.
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit/.
// SKELETON: exact CPU single-layer-test fixture symbols must be confirmed against the
//           intel_cpu/tests/unit tree before use.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"

using namespace ov;

// TODO: confirm the intel_cpu unit-test fixture base / helper (e.g. the test
// helpers under src/plugins/intel_cpu/tests/unit/) and reuse it instead of raw ov::Core.
TEST(CpuPadNegativePadOverflow, InnerCopySizeWraparoundRejected) {
    // Data input: dynamic rank-1 shape so pads are data-dependent (dynamic path).
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto pads_begin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pads_end   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    // v1 Pad, CONSTANT mode (no begin/end-vs-dim validation in shape_infer for CONSTANT).
    auto pad = std::make_shared<op::v1::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad},
                                         ParameterVector{data, pads_begin, pads_end});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // srcDim = 3, padsBegin = -100, padsEnd = +200  -> output dim 103, innerCopySize wraps.
    Tensor t_data(element::f32, Shape{3});
    std::fill_n(t_data.data<float>(), 3, 0.0f);
    Tensor t_begin(element::i32, Shape{1}); t_begin.data<int32_t>()[0] = -100;
    Tensor t_end(element::i32, Shape{1});   t_end.data<int32_t>()[0]   =  200;
    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_begin);
    req.set_input_tensor(2, t_end);

    // Pre-fix: ASan reports heap-buffer-overflow inside cpu_memcpy (pad.cpp:528/574).
    // Post-fix: PadExecutor ctor throws ov::Exception from innerParamsInitialization.
    EXPECT_ANY_THROW(req.infer());
}
