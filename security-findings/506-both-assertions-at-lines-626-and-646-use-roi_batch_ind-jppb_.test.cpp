// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for roi_pooling.cpp:626 / 646 (OOB read off-by-one).
// Pre-fix: roi_batch_ind == b_num (e.g. featureShape[0]==2, ROI batch index 2.0f)
//   passes the `<= jpp.b_num` assertion and the JIT path indexes one batch-stride
//   past src_data -> ASan heap-buffer-overflow READ.
// Post-fix: the tightened `< jpp.b_num` assertion makes ROIPooling throw on the
//   out-of-range batch index, so this test expects an exception.
//
// Harness: ov_cpu_unit_tests (intel_cpu). TODO: confirm exact test dir/style under
//   src/plugins/intel_cpu/tests/unit/ and the helper that builds a single-node graph.
#include <gtest/gtest.h>
// TODO: include the intel_cpu single-node test helpers (e.g. the graph/test_utils
//       headers used by existing nodes/* unit tests) — read tests/unit to get exact names.

TEST(ROIPoolingOOB, RoiBatchIndEqualsBatchCountIsRejected) {
    // TODO: build a ROIPooling node where feature input has batch N == 2
    //       (featureShape[0] == 2 -> jpp.b_num == 2), bilinear or max alg, on CPU.
    // TODO: feed a ROI tensor whose first element (roi_batch_ind) == 2.0f.
    //   const float roi[5] = {2.0f, 0.f, 0.f, 1.f, 1.f};
    // Pre-fix this passes the `<=` assert and reads OOB (ASan abort);
    // post-fix the node must throw on the out-of-range batch index.
    EXPECT_ANY_THROW({
        // TODO: run inference on the assembled single-ROIPooling graph with the
        //       crafted ROI tensor above.
    });
}
