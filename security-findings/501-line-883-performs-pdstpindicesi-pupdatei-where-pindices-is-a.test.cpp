// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:882-885 (missing bounds check on the
// 1-D i32 short-vector ScatterUpdate fast path). Pre-fix: pdst[pindices[i]]
// with an out-of-range index performs an OOB heap write (ASan heap-buffer-overflow).
// Post-fix: the node must reject the out-of-bounds index via CPU_NODE_ASSERT
// (ov::Exception) instead of writing past the destination buffer.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). SKELETON — exact graph/
// infer-request helper symbols must be confirmed against the existing unit tree
// before use.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"

using namespace ov;

TEST(ScatterUpdateCpu1DFastPath, OutOfRangeIndexIsRejected) {
    // TODO: confirm correct test fixture/base in intel_cpu/tests/unit and the
    //       canonical way to build+compile a single-op model on the CPU plugin.

    // Build: data[4] (i32), indices[1] (i32), axis const(0), update[1] (i32)
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto update  = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, update, axis);
    auto model = std::make_shared<Model>(OutputVector{su->output(0)},
                                         ParameterVector{data, indices, update});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::i32, Shape{4});
    std::fill_n(t_data.data<int32_t>(), 4, 0);
    Tensor t_idx(element::i32, Shape{1});
    t_idx.data<int32_t>()[0] = 200;   // out of [0,4) -> OOB write pre-fix
    Tensor t_upd(element::i32, Shape{1});
    t_upd.data<int32_t>()[0] = 0x41414141;

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_idx);
    req.set_input_tensor(2, t_upd);

    // Pre-fix: ASan reports heap-buffer-overflow on pdst[200] at scatter_update.cpp:883.
    // Post-fix: index is range-checked and infer() throws ov::Exception.
    EXPECT_ANY_THROW(req.infer());

    // TODO: also add a negative-index case (t_idx=-1) which writes before the buffer.
}