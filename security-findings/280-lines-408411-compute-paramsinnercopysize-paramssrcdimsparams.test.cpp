// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408-411
// (and the missing upstream magnitude validation). Pre-fix: a Pad op in CONSTANT
// mode with dynamic pads_begin=-13, pads_end=+13 on a srcDim=3 axis yields a valid
// output shape (3) but innerCopySize wraps to ~SIZE_MAX, driving cpu_memcpy
// (pad.cpp:528/574/619) far past the heap allocation -> ASan heap-buffer-overflow.
// Post-fix: the over-crop is rejected (throw) OR innerCopySize is clamped to 0 so
// no OOB occurs and inference completes deterministically.
//
// TODO(symbols): confirm the exact single-layer test base class / helpers used in
// openvino/src/plugins/intel_cpu/tests/functional (e.g. SubgraphBaseTest / ov::Model
// builder) — names below are best-effort and must be checked against the tree.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

// Build: data[1,1,3] padded on last axis with DYNAMIC pads inputs begin=-13,end=+13.
TEST(intel_cpu_Pad, NegativeCropExceedingDim_NoHeapOverflow) {
    using namespace ov;
    auto data   = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 3});
    auto pbegin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    auto pend   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    auto pval   = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto pad    = std::make_shared<op::v1::Pad>(data, pbegin, pend, pval, op::PadMode::CONSTANT);
    auto model  = std::make_shared<Model>(OutputVector{pad}, ParameterVector{data, pbegin, pend});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float>   in_data(3, 1.0f);
    std::vector<int32_t> begin{0, 0, -13};
    std::vector<int32_t> end{0, 0, 13};
    req.set_input_tensor(0, Tensor(element::f32, Shape{1,1,3}, in_data.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{3}, begin.data()));
    req.set_input_tensor(2, Tensor(element::i32, Shape{3}, end.data()));

    // Pre-fix this infer triggers an ASan heap-buffer-overflow inside cpu_memcpy.
    // Post-fix either the over-crop is rejected, or it runs without OOB.
    // Accept either a clean throw (validation added) or a clean completion (clamp).
    try {
        req.infer();
        SUCCEED() << "completed without heap overflow (clamp path)";
    } catch (const ov::Exception&) {
        SUCCEED() << "rejected over-crop via validation (preferred fix)";
    }
    // TODO: under ASan, the assertion of interest is simply that the process does
    // NOT abort with heap-buffer-overflow; gtest's death is observed by the harness.
}
