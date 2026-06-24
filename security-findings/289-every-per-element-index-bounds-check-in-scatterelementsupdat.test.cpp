// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:605/627/650/669 (and ReduceMean :734/758/793/814).
// Pre-fix: a ScatterElementsUpdate whose indices contain a value more negative than
// -data_shape[axis] (e.g. -(dim+1000)) passes the upper-bound check at :913
// (negatives are allowed for ScatterElementsUpdate) and, because the per-element
// assert() is compiled out under NDEBUG, normalization (idxValue += data_dim_size)
// leaves a negative offset -> OOB heap write at :606/628 (ASan: heap-buffer-overflow
// / SEGV). Post-fix the CPU_NODE_ASSERT must reject the model with ov::Exception.
//
// Harness: ov_cpu_unit_tests (intel_cpu component, gtest). SKELETON — exact graph-build
// helpers / ScatterElementsUpdate op factory and the CPU graph-exec wrapper used by
// intel_cpu/tests/unit must be confirmed by reading the neighboring tests.
#include <gtest/gtest.h>
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/runtime/core.hpp"

TEST(scatter_elements_update_cpu, negative_index_below_neg_dim_is_rejected) {
    // TODO: build a tiny ov::Model with a v12 ScatterElementsUpdate node:
    //   data    : shape {4}, f32
    //   indices : shape {1}, i64, VALUE = -(4 + 1000)  // out of [-4,3]
    //   updates : shape {1}, f32
    //   axis    : 0
    // TODO: compile on "CPU" and infer; the crafted negative index must be rejected.
    // EXPECT_THROW(run_on_cpu(model, inputs), ov::Exception);   // passes only after fix
    GTEST_SKIP() << "TODO: wire up intel_cpu graph-exec helper + ScatterElementsUpdate factory";
}