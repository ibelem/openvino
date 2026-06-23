// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in GatherND::GatherNDExecutor::gatherElementwise
// Unchecked index path: gather_nd.cpp:256-259 (dataIdx via HandleNegativeIndices, gather_nd.cpp:272-278).
// Pre-fix: with dataLength==1 (scalar slice) and an indices value >= the data dimension,
//          shiftedSrcData[dataIdx] reads past the source buffer -> ASan heap-buffer-overflow.
// Post-fix: HandleNegativeIndices throws ov::Exception for out-of-range index, so inference
//           must reject the bad input (ASSERT_ANY_THROW) instead of reading OOB.
//
// NOTE: GatherNDExecutor and HandleNegativeIndices are PRIVATE nested members of
// ov::intel_cpu::node::GatherND and cannot be instantiated directly from the test
// harness. The flaw must therefore be exercised end-to-end by compiling a GatherND
// subgraph on the CPU plugin and running inference with a crafted out-of-range index.
// This is a SKELETON: the exact subgraph-builder / infer helpers must be filled from the
// surrounding ov_cpu_unit_tests tree (e.g. nodes/*_node_test.cpp) before use.

#include <gtest/gtest.h>
#include <vector>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(GatherND_CPU_OOB, ElementwiseIndexOutOfBoundsIsRejected) {
    // data shape [5] -> dataLength == 1 so gatherElementwise() path is taken.
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    // indices select a single scalar element; value 200 is far out of range for dim 5.
    // TODO: confirm indices precision/shape accepted by GatherND v8 (sliceRank == last dim == 1).
    auto indices = op::v0::Constant::create(element::i32, Shape{1, 1}, std::vector<int32_t>{200});
    auto gnd = std::make_shared<op::v8::GatherND>(data, indices, /*batch_dims=*/0);
    auto model = std::make_shared<Model>(OutputVector{gnd}, ParameterVector{data}, "gathernd_oob");

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{5});
    auto* p = in.data<int32_t>();
    for (int i = 0; i < 5; ++i) p[i] = i;
    req.set_input_tensor(in);

    // Pre-fix: ASan flags heap-buffer-overflow inside gatherElementwise (gather_nd.cpp:259).
    // Post-fix: HandleNegativeIndices throws ov::Exception, surfaced here as a thrown exception.
    ASSERT_ANY_THROW(req.infer());
}