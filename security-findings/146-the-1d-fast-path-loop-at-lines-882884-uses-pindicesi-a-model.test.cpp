// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:882-884 (ScatterUpdate 1D fast-path OOB write).
// Pre-fix: an out-of-range INDICES value drives pdst[pindices[i]]=pupdate[i] past the
//          dst allocation -> ASan heap-buffer-overflow (or silent corruption).
// Post-fix: the mirrored CPU_NODE_ASSERT(pindices[i] in [0,srcDataDim[0])) rejects it,
//          surfacing as an ov::Exception at inference time.
//
// NOTE: ov_cpu_unit_tests is primarily a node-level harness; reaching the fast-path
// cleanly is easiest via a compiled CPU model that exercises opset3 ScatterUpdate with
// a 1D i32 DATA of length <= 64 and a malicious INDICES constant. Symbols below
// (ov::Core, opset3) are standard; the exact include/test-registration may need tuning
// against the surrounding tree, hence this is a SKELETON.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset3.hpp"

using namespace ov;

TEST(ScatterUpdateCpu1DFastPath, RejectsOutOfRangeIndex) {
    // DATA: 1D i32, length 16 (<=64) -> satisfies fast-path predicate at scatter_update.cpp:868-869
    auto data = std::make_shared<opset3::Parameter>(element::i32, Shape{16});
    // INDICES: single i32 index = 17 (>= srcDataDim[0]==16) -> OOB write pre-fix
    auto indices = opset3::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{17});
    auto axis    = opset3::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto updates = opset3::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{0x7fffffff});

    auto su = std::make_shared<opset3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(NodeVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{16});
    std::fill_n(in.data<int32_t>(), 16, 0);
    req.set_input_tensor(in);

    // TODO: if the fast-path requires a const-folded DATA path or shape-inference subgraph
    //       context to be selected (see comment "optimized for shape inference subgraph"),
    //       wrap DATA accordingly so exec1DCase is chosen.
    EXPECT_THROW(req.infer(), ov::Exception);
}
