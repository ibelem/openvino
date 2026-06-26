// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-129/787 at openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:913-628.
// Pre-fix: a ScatterElementsUpdate node whose INDICES contain a value < -data_dim_size
// (e.g. -1000 on axis-0 size 10) passes the CPU_NODE_ASSERT at line 913 (OR-clause) and the
// release-stripped assert at line 627, then computes dataPtr[offsets[0] + int64_t(-990)*block]
// where the negative int64_t is promoted to a huge uint64_t => heap OOB write (ASan: heap-buffer-overflow WRITE).
// Post-fix: the node must reject/throw on out-of-range negative indices, OR enforce idxValue>=-data_dim_size.
//
// SKELETON: the exact intel_cpu unit-test fixture symbols (ov_cpu_unit_tests node-test helpers)
// must be confirmed against intel_cpu/tests/unit before this will compile.
#include <gtest/gtest.h>
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/runtime/core.hpp"

// TODO: confirm correct test target & include layout under
//       openvino/src/plugins/intel_cpu/tests/unit/ — symbol names below are placeholders.
TEST(ScatterElementsUpdateCpu, RejectsDeeplyNegativeIndex) {
    using namespace ov;
    // data: shape [10], axis 0; indices: [-1000]; updates: [42]
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto indices = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1000});
    auto updates = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{42.0f});
    auto axis    = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    Tensor in(element::f32, Shape{10});
    std::fill_n(in.data<float>(), 10, 0.0f);
    req.set_input_tensor(in);
    // Pre-fix this performs an OOB write (ASan abort). Post-fix the node must reject the index.
    ASSERT_ANY_THROW(req.infer());
}
