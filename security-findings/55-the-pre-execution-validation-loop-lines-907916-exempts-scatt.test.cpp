// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:913-915 (validation exemption)
//   and the kernel deref at line 606 (assert is a no-op under NDEBUG).
//
// Encodes: a ScatterElementsUpdate model whose indices tensor contains a value
// < -data_dim_size (e.g. -(N+1) with N=axis-dim) must be REJECTED at execution
// (CPU_NODE_ASSERT -> ov::Exception), not silently dereferenced. Pre-fix this
// passes the validation loop, idxValue normalizes to -1, and ASan reports a
// heap-buffer-overflow write before the data buffer at scatter_update.cpp:606.
//
// Harness: ov_cpu_unit_tests (component target for openvino/src/plugins/intel_cpu).
// SKELETON: exact graph-build / infer-request helper symbols must be copied from
// the nearest existing tests under intel_cpu/tests/unit/ (e.g. the subgraph/
// single-layer test fixtures) — they are not guessed here.

#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test helpers actually used by the existing
//       tests in src/plugins/intel_cpu/tests/unit/ (graph builder + infer harness).
//       e.g. #include "nodes/..." or the test graph utility headers — READ that
//       dir to find the real fixture base class and helper names.

#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(ScatterElementsUpdate_CPU, RejectsIndexBelowNegativeDimSize) {
    // data: shape [10], axis = 0  -> data_dim_size = 10
    constexpr int64_t N = 10;
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{static_cast<size_t>(N)});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float>   dbuf(N, 0.0f);
    std::vector<int32_t> ibuf{ static_cast<int32_t>(-N - 1) }; // -11: passes buggy check, normalizes to -1
    std::vector<float>   ubuf{ 1.0f };

    req.set_input_tensor(0, Tensor(element::f32, Shape{static_cast<size_t>(N)}, dbuf.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{1}, ibuf.data()));
    req.set_input_tensor(2, Tensor(element::f32, Shape{1}, ubuf.data()));

    // Post-fix: out-of-range index is rejected before the kernel deref.
    // Pre-fix (release/NDEBUG): no throw and ASan reports heap-buffer-overflow WRITE
    //   at scatter_update.cpp:606.
    ASSERT_ANY_THROW(req.infer());
}
