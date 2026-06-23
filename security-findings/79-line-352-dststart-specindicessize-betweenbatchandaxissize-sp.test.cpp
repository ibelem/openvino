// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 in openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:352
// (also :354/:359). Pre-fix: a static Gather with a zero-element indices tensor makes
// specIndicesSize==0 (accumulate at gather.cpp:182), and createPrimitive's parallel_nt
// lambda performs `dstStart / specIndicesSize` == 0/0 -> SIGFPE.
// Post-fix: the `if (specIndicesSize==0 || afterAxisSize==0) return;` guard skips the JIT
// param precompute and the model compiles without crashing (empty output via execReference).
//
// HARNESS: ov_cpu_unit_tests (intel_cpu). This is a SKELETON: the exact CPU-node test
// helpers/symbols were not read, so fill in the TODOs against the real test tree before use.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm the correct fixture/namespace by reading the nearest existing tests under
//       openvino/src/plugins/intel_cpu/tests/unit/ (e.g. how a Model is compiled for "CPU").
TEST(GatherCpuDivZero, EmptyIndicesDoesNotCrashOnCompile) {
    using namespace ov;

    // data: static, non-trivial after-axis collapses to afterAxisSize==1 so JIT path is taken.
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    // indices: STATIC zero-element tensor -> specIndicesSize == product(idxDims) == 0.
    auto indices = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    Core core;
    // Pre-fix this aborts with SIGFPE inside Gather::createPrimitive (gather.cpp:352).
    // Post-fix it must compile cleanly (no throw, no crash).
    // TODO: if empty-shape Gather is rejected earlier by shape inference on this build,
    //       use a [N,0] indices Constant instead so product==0 while rank>0.
    EXPECT_NO_THROW({ auto compiled = core.compile_model(model, "CPU"); });
}
