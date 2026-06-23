// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read/write in Gather::execCompressed4Bit
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:735-737
// Pre-fix: with afterAxisSize % scale_group_size != 0 the inner loop
//   `for (g = p; g < p + scale_group_size; g++)` overruns the per-row source
//   slice (srcData[g>>1], line 736) and destination slice (pdst[dst_idx],
//   line 737); on the final slice this is a heap OOB read + write that ASan
//   reports. Post-fix (loop clamped to min(p+scale_group_size, srcIdx+afterAxisSize)
//   and/or divisibility validation) the model is rejected or runs in-bounds.
//
// SKELETON: building a GatherCompressed internal op + running CPU inference
// requires the intel_cpu graph test fixtures; exact symbols must be taken from
// the surrounding test tree before this will compile.
//
// Harness: ov_cpu_unit_tests (gtest + ASan), style of
//   targets/openvino/src/plugins/intel_cpu/tests/unit/

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
// TODO: include the CPU-plugin graph/infer harness headers used by the
//       existing tests under intel_cpu/tests/unit/ (e.g. the helper that
//       compiles an ov::Model on the CPU device and runs a single infer).

using namespace ov;

TEST(GatherCompressed4Bit, NonDivisibleGroupSizeIsRejectedOrInBounds) {
    // data: u4 [axisDim=2, afterAxisSize=7]  (7 nibbles per row)
    // scale: f32 [2, 3] -> scale_group_size = (2*7)/(2*3) = 2, and 7 % 2 == 1
    // indices: [0] on axis 0 -> selects a full 7-element row, drives the
    //          (cond1||cond2) grouped path with non-scalar scale (cond3 false).
    // TODO: construct the parameters/constants with the exact builder helpers
    //       used by the CPU unit tests:
    //   auto data    = <u4 Parameter/Constant, shape {2,7}>;
    //   auto indices = op::v0::Constant({0});           // axis-0 row select
    //   auto axis    = op::v0::Constant(0);
    //   auto scale   = <f32 Constant, shape {2,3}>;     // non-divisible group
    //   auto gc = std::make_shared<op::internal::GatherCompressed>(
    //                 data, indices, axis, /*batch_dims=*/0, scale);
    //   auto model = std::make_shared<ov::Model>(gc, ParameterVector{...});
    //
    // Pre-fix expectation: compiling+infer on CPU triggers ASan
    //   heap-buffer-overflow at gather.cpp:736/737.
    // Post-fix expectation: either the malformed group size is rejected at
    //   compile time (ov::Exception) or inference completes in-bounds.
    //
    // TODO: replace with the harness call, e.g.:
    //   EXPECT_NO_THROW({ auto compiled = core.compile_model(model, "CPU");
    //                     auto req = compiled.create_infer_request();
    //                     req.infer(); });
    // (the real assertion is that ASan does NOT fire; if the chosen fix
    //  rejects malformed shapes instead, switch to EXPECT_THROW(..., ov::Exception)).
    GTEST_SKIP() << "TODO: wire up CPU compile_model/infer harness and u4 data init";
}
