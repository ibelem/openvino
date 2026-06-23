// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987  (Gather::exec1DCase)
// Unchecked line: `pdst[i] = psrc[ii];` with no `ii < axisDim` guard
// (contrast Gather::execReference gather.cpp:947 which clamps).
//
// What this encodes: drive the intel_cpu Gather 1D-optimized path
// (dataSrcRank<=1, i32 data, 1D data & index dims <=64) with an index whose
// value is >= axisDim (and a negative index with reverseIndexing=false).
// PRE-FIX: ASan reports a heap-buffer-overflow READ in Gather::exec1DCase,
// and/or the output element equals leaked adjacent heap memory.
// POST-FIX: the out-of-range lanes are zeroed, so the output is deterministic
// (== 0) and ASan is clean.
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan). Place under
//   openvino/src/plugins/intel_cpu/tests/unit/  next to the existing node
//   single-layer unit tests.
//
// NOTE: emitted as a SKELETON — the exact intel_cpu single-node unit-test
// fixture symbols were not confirmed by reading the test tree, and the 1D
// path requires constructing a real Gather node with a graph/edge context.

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// TODO: confirm the canonical intel_cpu node-level unit-test fixture name and
//       includes (e.g. the helpers used by gather node tests under
//       intel_cpu/tests/unit/). This functional-style variant uses ov::Core +
//       CPU device to force the i32 1D optimized path.
TEST(intel_cpu_gather_exec1DCase, oob_index_must_be_clamped_not_read) {
    // axisDim = 4, 1D i32 data -> hits canOptimize1DCase (prepareParams:396-401)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // Out-of-range index value 1000000 (>= axisDim) AND -1 (negative).
    auto indices = op::v0::Constant::create(element::i32, Shape{2}, {1000000, -1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    // reverseIndexing defaults false for v8 Gather batch_dims=0.
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, 0);
    auto model   = std::make_shared<Model>(OutputVector{gather->output(0)},
                                           ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{4});
    auto* p = in.data<int32_t>();
    p[0] = 10; p[1] = 11; p[2] = 12; p[3] = 13;
    req.set_input_tensor(in);

    // PRE-FIX: ASan traps inside Gather::exec1DCase reading psrc[1000000]/psrc[4].
    // POST-FIX: out-of-range lanes are zeroed (matches execReference).
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor();
    const auto* o = out.data<int32_t>();
    EXPECT_EQ(o[0], 0);  // index 1000000 -> out of range -> 0
    EXPECT_EQ(o[1], 0);  // index -1, reverseIndexing=false -> axisDim -> 0
}