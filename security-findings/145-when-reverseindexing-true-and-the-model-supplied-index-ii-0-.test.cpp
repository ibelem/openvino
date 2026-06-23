// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:982-987 (Gather::exec1DCase)
// Pre-fix: a 1-D i32 Gather (data shape [1], indices [-200], reverseIndexing=true)
//          drives canOptimize1DCase==true (prepareParams:399-401) and execute()
//          dispatches to exec1DCase(), where `ii += axisDim` leaves ii negative and
//          `psrc[ii]` reads before the source allocation -> ASan heap-buffer-overflow.
// Post-fix: the added bounds guard zero-fills the out-of-range element; inference
//           succeeds and the output element equals 0 (matches execReference semantics).
//
// HARNESS: ov_cpu_unit_tests (the intel_cpu component's own unit target;
//          inferred from src/plugins/intel_cpu/tests/unit/CMakeLists.txt:7).
// NOTE: exec1DCase() is private and only reachable by building/executing a Gather
//       subgraph, so this is a best-effort SKELETON driven through the public
//       ov::Core / ov::InferRequest API. // TODO items mark the pieces that must be
//       confirmed against the real headers before this will compile.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm the correct fixture/base used by intel_cpu unit tests for
//       end-to-end CPU inference (read tests/unit/ for an existing example that
//       calls core.compile_model(model, "CPU")).
TEST(GatherExec1DCase, NegativeIndexBeyondAxisDimDoesNotReadOOB) {
    using namespace ov;

    // 1-D i32 data of shape [1] -> axisDim = 1 (<=64 enables canOptimize1DCase).
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // indices = [-200]; |index| > axisDim so ii stays negative after ii += axisDim.
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-200});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    // v8 Gather -> reverseIndexing defaults to true (gather.cpp:115).
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims*/0);
    auto model   = std::make_shared<Model>(OutputVector{gather->output(0)},
                                           ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{1});
    in.data<int32_t>()[0] = 42;
    req.set_input_tensor(in);

    // Pre-fix: ASan aborts here with heap-buffer-overflow (READ) inside exec1DCase.
    ASSERT_NO_THROW(req.infer());

    // Post-fix: out-of-range index is zero-filled (matches execReference 959-963).
    auto out = req.get_output_tensor();
    EXPECT_EQ(out.data<int32_t>()[0], 0);
}
