// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 / CWE-369 in
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:79
//   int ic_off = IC / (stride * stride);   // signed 32-bit overflow for stride>=46341,
//                                          // and ic_off==0 -> modulo-by-zero at line 88
// Pre-fix: building/executing a ReorgYolo with a huge stride (e.g. 50000) on a
//   dynamic-channel input whose concrete C is small triggers UB / SIGFPE (or an
//   out-of-bounds read flagged by ASan) inside ReorgYolo::execute.
// Post-fix: the node must reject the out-of-range stride / zero ic_off and throw
//   ov::Exception instead of executing.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). The exact single-node test
// scaffolding (graph-builder helper, infer-request driver) varies across the
// CPU unit-test tree, so this is emitted as a SKELETON.

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(ReorgYoloCpu, RejectsOverflowingStride) {
    // TODO: confirm the canonical single-op CPU test helper in
    //       src/plugins/intel_cpu/tests/unit/ (e.g. a fixture that compiles a
    //       Model on the "CPU" device and runs infer). Replace the manual
    //       core.compile_model below with that helper if one exists.

    // Dynamic channel dimension so reorg_yolo_shape_inference.hpp:35 skips the
    // C >= stride*stride validation and the model is constructible.
    auto input = std::make_shared<op::v0::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), 8, 8});

    // stride = 50000 -> 50000*50000 overflows int32 at reorg_yolo.cpp:79.
    const size_t kOverflowStride = 50000;
    auto reorg = std::make_shared<op::v0::ReorgYolo>(input, Strides{kOverflowStride, kOverflowStride});
    auto result = std::make_shared<op::v0::Result>(reorg);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // Concrete small C (=4) << stride*stride forces ic_off==0 / overflow at
    // execute time.
    Tensor in(element::f32, Shape{1, 4, 8, 8});
    req.set_input_tensor(in);

    // TODO: pre-fix this either SIGFPEs / ASan-aborts (no clean throw) — confirm
    //       whether the post-fix surfaces ov::Exception from infer(); adjust the
    //       matcher accordingly.
    EXPECT_ANY_THROW(req.infer());
}
