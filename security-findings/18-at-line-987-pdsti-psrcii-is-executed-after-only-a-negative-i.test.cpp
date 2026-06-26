// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 OOB read in Gather::exec1DCase
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no upper-bound check on positive user index.
// The 1D fast path is selected in prepareParams (gather.cpp:396-402) whenever
// the data input is rank<=1, i32, dim<=64 and indices rank<=1, dim<=64.
// Feeding an index >= axisDim must NOT read past the source buffer. Pre-fix
// this triggers an ASan heap-buffer-overflow read; post-fix the out-of-range
// element is zero-filled (consistent with execCompressed gather.cpp:960-962).
//
// Harness: ov_cpu_unit_tests (gtest + ASan), file under intel_cpu/tests/unit/nodes/.
//
// SKELETON: exec1DCase() is a private method and requires a fully wired Node
// graph + allocated MemoryPtr inputs/outputs to reach. Building that inline is
// non-trivial; the cleanest reproduction is a small ov::Model subgraph compiled
// on CPU and run with a crafted indices tensor. TODOs mark the missing pieces.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov;

// TODO: confirm the include path / fixture base used by neighbouring tests in
//       intel_cpu/tests/unit/nodes/ (read an existing *_node_test.cpp first).

TEST(GatherExec1DCaseOOB, PositiveIndexOutOfRangeIsZeroFilledNotOOB) {
    // Build: data[N] i32 (N<=64), indices[M] i32 (M<=64)  -> triggers 1D fast path.
    constexpr size_t N = 10;
    auto data = std::make_shared<opset8::Parameter>(element::i32, Shape{N});
    auto indices = std::make_shared<opset8::Parameter>(element::i32, Shape{1});
    auto axis = opset8::Constant::create(element::i32, Shape{}, {0});
    auto gather = std::make_shared<opset8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{gather},
                                         ParameterVector{data, indices});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> dataVals(N);
    for (size_t i = 0; i < N; ++i) dataVals[i] = static_cast<int32_t>(i);
    Tensor dataT(element::i32, Shape{N}, dataVals.data());

    // Out-of-range positive index: 1000 >> axisDim (10).
    int32_t badIdx = 1000;
    Tensor idxT(element::i32, Shape{1}, &badIdx);

    req.set_input_tensor(0, dataT);
    req.set_input_tensor(1, idxT);

    // Pre-fix: exec1DCase reads psrc[1000] -> ASan heap-buffer-overflow READ.
    // Post-fix: out-of-range index is zero-filled; infer completes cleanly.
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor(0);
    EXPECT_EQ(out.data<int32_t>()[0], 0);  // matches execCompressed zero-fill semantics

    // TODO: if the operator contract instead mandates throwing on OOB indices,
    //       replace ASSERT_NO_THROW with EXPECT_THROW(req.infer(), ov::Exception)
    //       and confirm against the chosen fix variant.
    // TODO: also add a negative-index, reverseIndexing==false case to cover the
    //       ii==axisDim off-by-one in gather.cpp:984.
}