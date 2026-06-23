// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in Gather::exec1DCase (openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:982-987).
// Pre-fix: indices value -100 with 1-D i32 data of shape [5] survives `ii += axisDim` as ii=-95,
//          then `psrc[ii]` (line 987) reads ~380 bytes before the data buffer -> ASan heap-buffer-overflow (read).
// Post-fix: the added `if (ii < 0 || (size_t)ii >= axisDim)` guard zero-fills the output, no OOB.
// Harness: ov_cpu_unit_tests (gtest + ASan). Place near intel_cpu/tests/unit/ Gather node tests.
//
// SKELETON — the exact intel_cpu unit-test fixture/helpers for constructing and executing a single
// Gather node graph were not read, so symbol names below are placeholders.
#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test graph/node helpers actually used under
//       src/plugins/intel_cpu/tests/unit/ (e.g. the test util that builds a CPU graph from an ov::Model
//       and runs infer). Read that tree to get the real headers and runner symbols.

#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(intel_cpu_Gather, exec1DCase_negative_index_below_minus_axisdim_no_oob) {
    // 1-D i32 data of shape [5], 1-D i32 indices of shape [1] = {-100}.
    // dims<=64 and i32 => prepareParams enables canOptimize1DCase (gather.cpp:399-401),
    // and v8::Gather without dontReverseIndices => reverseIndexing==true (gather.cpp:115).
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{-100});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model   = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    // TODO: build the CPU graph for `model` and run infer with input data {0,1,2,3,4}
    //       using the intel_cpu unit-test runner. With the fix the single output element
    //       must be 0 (out-of-range index -> zero-fill, matching execReference at line 947-963).
    //
    //   auto out = run_cpu_single_infer(model, /*input i32[5]=*/{0,1,2,3,4});
    //   ASSERT_EQ(out.size(), 1u);
    //   EXPECT_EQ(out[0], 0);   // fix zero-fills; pre-fix this line is never reached (ASan abort earlier)
    GTEST_SKIP() << "TODO: wire intel_cpu unit-test graph runner; see gather.cpp:982-987";
}