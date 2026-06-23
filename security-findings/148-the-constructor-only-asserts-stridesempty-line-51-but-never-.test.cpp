// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero in ReorgYolo.
// Encodes the fix cited at:
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:52 (constructor: missing stride>0 check)
//   triggering the SIGFPE at reorg_yolo.cpp:79 `IC / (stride*stride)`.
// Pre-fix: compiling/running a ReorgYolo with stride=0 on the CPU plugin reaches
//          `IC / 0` -> SIGFPE (or, with the fix in the core op, an upstream throw).
// Post-fix: the CPU node constructor's CPU_NODE_ASSERT(stride>0, ...) makes
//          compile_model on the CPU device throw ov::Exception before execution.
//
// NOTE: ReorgYolo::execute needs allocated edges/memory, so the cleanest harness
// is to build a tiny ov::Model and compile it on the CPU plugin and assert the
// throw. Symbols/headers below were modelled on the intel_cpu unit test tree but
// MUST be reviewed for exact include paths and target wiring.
//
// TODO: confirm test file lands under src/plugins/intel_cpu/tests/unit/nodes/
//       and is picked up by the ov_cpu_unit_tests glob (see tests/unit/CMakeLists.txt).
#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/runtime/core.hpp"

// stride=0 must be rejected (post-fix) rather than crash with SIGFPE (pre-fix).
TEST(ReorgYoloNodeTest, StrideZeroIsRejectedOnCpu) {
    using namespace ov;

    // [N, C, H, W] with C divisible setup is irrelevant once stride==0.
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 4, 6, 6});

    // stride attribute = 0 (attacker-controlled in a crafted model).
    // TODO: verify the Strides ctor overload; op::v0::ReorgYolo(input, Strides{0,0}).
    auto reorg = std::make_shared<op::v0::ReorgYolo>(param->output(0), Strides{0, 0});
    auto result = std::make_shared<op::v0::Result>(reorg->output(0));
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    Core core;
    // With the constructor fix, CPU node creation throws during compile_model.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
