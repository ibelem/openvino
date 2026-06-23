// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp
//   - exemption at lines 913-915 (execute pre-check skips lower-bound for ScatterElementsUpdate)
//   - single-step normalization 602-603/624-625/647-648/666-667/731-732/755-756/790-791/811-812
//   - dead asserts at 605/627/650/669/734/758/793/814 (no-op under NDEBUG)
//
// Intent: feed a ScatterElementsUpdate op an index value of -(data_dim_size+1)
// on the scatter axis. Pre-fix: index normalizes to -1, the dead assert is a
// no-op in release, and dataPtr[offsets[0] + (-1)*dataBlock_axisplus1] writes
// out of bounds (ASan: heap-buffer-overflow WRITE). Post-fix: an active
// CPU_NODE_ASSERT after normalization rejects the index -> ov::Exception.
//
// TODO(verify): exact target/symbols. File lives under intel_cpu, so the
// harness is `ov_cpu_unit_tests`. This is a SKELETON: the direct node API
// (ScatterUpdate node) is not trivially constructible standalone, so the test
// drives it via an ov::Model + CPU plugin infer request, which is the pattern
// used by intel_cpu single-layer tests. Confirm helper/include names against
// the existing intel_cpu test tree before use.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// TODO: confirm CPU device is registered as "CPU" in this test target's setup.
TEST(scatter_elements_update_cpu, negative_index_below_neg_dim_is_rejected) {
    // data: shape [4] -> data_dim_size on axis 0 is 4
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});

    // indices: a single out-of-range negative value -(4+1) = -5 on axis 0.
    // Pre-fix normalization gives -5 + 4 = -1 (still negative) -> OOB write.
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-5});
    auto updates = op::v0::Constant::create(element::f32, Shape{1}, {42.0f});
    auto axis    = op::v0::Constant::create(element::i32, Shape{1}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor input(element::f32, Shape{4});
    auto* p = input.data<float>();
    for (size_t i = 0; i < 4; ++i) p[i] = 1.0f;
    req.set_input_tensor(input);

    // Post-fix: kernel rejects the unnormalizable negative index.
    // Pre-fix: ASan flags heap-buffer-overflow WRITE inside scatterElementsUpdate.
    EXPECT_ANY_THROW(req.infer());
}
