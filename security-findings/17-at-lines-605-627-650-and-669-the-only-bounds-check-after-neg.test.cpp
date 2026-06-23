// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:602-670
// (and the ReduceMean specialization at 700-815).
//
// Encodes the fix: a ScatterElementsUpdate whose indices tensor contains a
// value more negative than -data_dim_size (e.g. -1000 for an axis of size 10)
// must be REJECTED with an ov::Exception (via CPU_NODE_ASSERT) instead of
// performing an out-of-bounds write. Pre-fix, in an NDEBUG/release build the
// bare assert() at line 605/627/650/669 is stripped and the negative index
// (normalized once to -990, still negative) flows into
//   dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]
// producing a wild store that ASan reports as heap-buffer-overflow.
//
// NOTE (why this is a SKELETON, not a turnkey test):
//  * The OOB only manifests under NDEBUG; a Debug-built ov_cpu_unit_tests will
//    instead abort on the surviving assert(), so the assertion semantics differ
//    by build type. The fix replaces assert() with CPU_NODE_ASSERT, which throws
//    in BOTH build types — that is what we assert below.
//  * The exact node-construction / infer-request helper symbols for driving a
//    single ScatterElementsUpdate op through ov_cpu_unit_tests must be copied
//    from the surrounding nodes/ test tree (e.g. the Subgraph/ngram node tests);
//    fill the TODOs after reading them.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_elements_update.hpp"

using namespace ov;

// TODO: confirm the fixture/helper used by neighbouring intel_cpu/tests/unit/nodes/*.cpp
// for compiling and infering a tiny single-op model on the CPU plugin.
TEST(ScatterElementsUpdateCPU, NegativeOutOfRangeIndexIsRejected) {
    // data: shape [10], axis 0  => data_dim_size = 10
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float> data_vals(10, 0.0f);
    std::vector<int32_t> idx_vals{-1000};  // |idx| >> data_dim_size, stays negative after +10
    std::vector<float> upd_vals{42.0f};

    req.set_input_tensor(0, Tensor(element::f32, Shape{10}, data_vals.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{1}, idx_vals.data()));
    req.set_input_tensor(2, Tensor(element::f32, Shape{1}, upd_vals.data()));

    // Post-fix: CPU_NODE_ASSERT converts the out-of-range index into a thrown
    // exception. Pre-fix (release/NDEBUG): silent OOB write (ASan heap-buffer-overflow).
    EXPECT_THROW(req.infer(), ov::Exception);
}
