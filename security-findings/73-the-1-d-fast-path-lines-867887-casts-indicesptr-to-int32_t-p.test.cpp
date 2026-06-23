// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-787 OOB write in
// openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883 (ScatterUpdate::execute 1-D fast path).
// Pre-fix: with data=[4] i32, indices=[1] i32 value=1000000 (or -1), the fast path executes
//   pdst[pindices[0]] = pupdate[0] with no bounds check -> ASan heap-buffer-overflow WRITE.
// Post-fix: the added CPU_NODE_ASSERT(idx>=0 && idx<srcDataDim[0]) rejects the model -> ov::Exception.
//
// TODO(harness): confirm exact target name (likely ov_cpu_unit_tests) and the model-building
//   helpers used under openvino/src/plugins/intel_cpu/tests/unit/ . This is a SKELETON: the CPU
//   node is normally driven via a compiled ov::Model + InferRequest, not constructed directly,
//   so the cleanest reproduction is an end-to-end infer on a tiny ScatterUpdate graph.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_scatter_update, oob_index_1d_fastpath_is_rejected) {
    // data: 1-D i32, <=64 elements -> triggers the fast-path gate (scatter_update.cpp:868-869)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // crafted out-of-range index (1000000) on the destination buffer of width 4
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {1000000});
    auto updates = op::v0::Constant::create(element::i32, Shape{1}, {0x41414141});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    Tensor in(element::i32, Shape{4});
    std::fill_n(in.data<int32_t>(), 4, 0);
    req.set_input_tensor(in);

    // Pre-fix: infer() performs an out-of-bounds heap write (ASan aborts here).
    // Post-fix: the bounds check throws before any write.
    EXPECT_THROW(req.infer(), ov::Exception);
    // TODO: if v3::ScatterUpdate constant-folds the index before reaching the CPU node,
    //   replace `indices`/`updates` with Parameters and feed the crafted index at runtime.
}