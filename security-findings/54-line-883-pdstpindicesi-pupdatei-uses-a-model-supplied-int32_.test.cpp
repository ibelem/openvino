// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for the OOB write at
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883
//   pdst[pindices[i]] = pupdate[i];   // model-supplied int32 index, no bounds check
// The 1-D fast path (lines 867-885) returns before the general-path index
// validation at lines 913-915, and the only guard (srcDataDim[0] <= 64, line 869)
// bounds the buffer size, not the index VALUE.
//
// This test builds a v3::ScatterUpdate model that hits the fast path
// (data i32 shape [4] <= 64, indices i32 shape [1], update i32 shape [1]) and
// feeds an out-of-range index value (1000). Pre-fix: ASan reports a heap OOB
// write inside ScatterUpdate::execute. Post-fix: the CPU_NODE_ASSERT throws an
// ov::Exception, so infer_request.infer() throws and ASSERT_ANY_THROW passes.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). Run on CPU plugin.
// TODO: confirm the exact include paths/helpers against a sibling test under
//       intel_cpu/tests/unit/ (e.g. an existing model-construction test) — the
//       symbol names below are best-effort and may need adjustment.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

TEST(ScatterUpdateCpuOOB, FastPathRejectsOutOfRangeIndex) {
    // data: i32 [4]  (<=64 -> enters the 1-D fast path)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // indices: i32 [1]
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // updates: i32 [1]
    auto updates = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // axis = 0
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> data_vals{0, 0, 0, 0};
    std::vector<int32_t> idx_vals{1000};   // out of range for a 4-element dst
    std::vector<int32_t> upd_vals{0x41414141};

    Tensor t_data(element::i32, Shape{4}, data_vals.data());
    Tensor t_idx(element::i32, Shape{1}, idx_vals.data());
    Tensor t_upd(element::i32, Shape{1}, upd_vals.data());

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_idx);
    req.set_input_tensor(2, t_upd);

    // Pre-fix: ASan aborts on heap-buffer-overflow at scatter_update.cpp:883.
    // Post-fix: CPU_NODE_ASSERT throws ov::Exception, surfaced by infer().
    ASSERT_ANY_THROW(req.infer());
}
