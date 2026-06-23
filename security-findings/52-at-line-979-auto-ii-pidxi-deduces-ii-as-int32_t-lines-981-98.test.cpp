// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in Gather::exec1DCase()
// Unchecked sink: openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no `ii >= 0 && ii < axisDim` guard after
//   the negative-index normalization at lines 980-986.
// Pre-fix: an index of INT32_MIN (or any value >= axisDim) on a 1-D i32
//   data tensor of <=64 elements drives exec1DCase() (selected by
//   prepareParams() lines 395-404) to read before/after the psrc buffer
//   -> ASan heap-buffer-overflow / SIGSEGV.
// Post-fix: the `idx < axisDim` guard zero-fills the OOB index, so the
//   graph executes safely and the output element is 0.
//
// HARNESS: ov_cpu_unit_tests / subgraph functional test (gtest).
// SKELETON: exact builder helpers + result-comparison macros must be
//   filled from the nearest existing Gather subgraph test before use.

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm namespace / fixture base used by intel_cpu unit tests
//       (e.g. ov::test::SubgraphBaseTest under intel_cpu/tests/unit).
TEST(GatherCpu1DCaseOobRead, NegativeIndexInt32MinIsRejectedNotOob) {
    using namespace ov;

    // 1-D i32 data of length <= 64 so prepareParams() enables exec1DCase.
    const std::vector<int32_t> dataVals(64, 7);
    auto data = std::make_shared<op::v0::Constant>(element::i32, Shape{64}, dataVals);

    // Indices input carrying the malicious INT32_MIN value.
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto axis = std::make_shared<op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    // v8::Gather -> reverseIndexing == true by default (gather.cpp:115).
    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{indices});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor idxTensor(element::i32, Shape{1});
    idxTensor.data<int32_t>()[0] = std::numeric_limits<int32_t>::min(); // INT32_MIN
    req.set_input_tensor(idxTensor);

    // Pre-fix: ASan reports heap-buffer-overflow inside exec1DCase here.
    // Post-fix: runs cleanly; out-of-range index zero-filled.
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor();
    EXPECT_EQ(out.data<int32_t>()[0], 0); // guard writes 0 for OOB index

    // TODO: verify get_output_tensor() element type matches outPrecision and
    //       adjust dtype accessor if the CPU plugin promotes i32 output.
}