# Security finding #506: Both assertions at lines 626 and 646 use `roi_batch_ind <= jpp.b_nu…

**Summary:** Both assertions at lines 626 and 646 use `roi_batch_ind <= jpp.b_nu…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read of up to one full batch worth of feature-map data (potentially megabytes). Under inference serving (e.g., an ONNX Runtime/OpenVINO EP deployment accepting user models), this can be triggered per-inference, enabling information disclosure of heap contents adjacent to the feature-map allocation, or a crash/DoS if the one-past-end region is unmapped.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/roi_pooling.cpp:626` — `ROIPooling::ROIPoolingJitExecutor::executeOptimizedGeneric()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ROI tensor input — the float value at src_roi[roi_off + 0] is attacker-controlled and crosses the trust boundary when the model/tensor is supplied by an untrusted caller

## Description / Root cause
Both assertions at lines 626 and 646 use `roi_batch_ind <= jpp.b_num` (inclusive upper bound). `jpp.b_num` is set to `featureShape[0]` (line 555), so valid batch indices are `[0, b_num-1]`. The `<=` allows `roi_batch_ind == b_num` to pass the assertion, after which the pointer arithmetic at line 667 (`src_data[roi_batch_ind * src_strides[0] + ...]`) and line 719 (`src_data[roi_batch_ind * src_strides[0] + ...]`) accesses one full batch-stride past the end of the allocated feature-map buffer.

**Validator analysis:** Confirmed off-by-one. Valid batch indices are [0,b_num-1] since b_num=featureShape[0] is the batch dimension size (roi_pooling.cpp:555). The assertions at 626/646 use `<=` rather than `<`, so a ROI whose first float casts to b_num (e.g. 2.0f when N=2) passes both checks and the pointer math at 667/719 (`roi_batch_ind*src_strides[0]+...`) addresses exactly one batch-stride past the end of src_data, an OOB read passed into the JIT kernel. CWE-125 is accurate; impact (info disclosure / DoS per-inference) is plausible. The proposed fix (change both `<=` to `<` and fix the message to print b_num-1) is correct and sufficient — it tightens the bound to the actual valid range. Note src_roi_ptr[0]==-1 (line 623) is an explicit sentinel handled before the assert; only the upper bound is buggy. Both repos validated: defect lives in openvino and is reachable via the EP's untrusted model/tensor path.

## Exploit / Proof of Concept
Supply a model whose feature input has batch size N (e.g. N=2). Provide a ROI tensor whose first element (roi_batch_ind field) is the float value 2.0f (== b_num). `static_cast<int>(2.0f)` yields 2. The assertion `0 <= 2 && 2 <= 2` passes at line 626 (pre-scan loop) and again at line 646 (parallel_for body). The code then computes `arg.src = &src_data[2 * src_strides[0] + ...]` — one full batch stride beyond the allocated buffer — and passes that pointer into the JIT kernel, triggering an OOB read of feature-map data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=ROIPoolingOOB.* . With ASan, the pre-fix code reports `heap-buffer-overflow READ` originating from ROIPooling::ROIPoolingJitExecutor::executeOptimizedGeneric (roi_pooling.cpp ~667/719) when roi_batch_ind==b_num; after changing `<=` to `<` the node throws and the test passes.

## Suggested fix
Change both assertions from `roi_batch_ind <= jpp.b_num` to `roi_batch_ind < jpp.b_num`:

  Line 626: `OPENVINO_ASSERT(0 <= roi_batch_ind && roi_batch_ind < jpp.b_num, ...);`
  Line 646: `OPENVINO_ASSERT(0 <= roi_batch_ind && roi_batch_ind < jpp.b_num, ...);`

Also update the error message from `"max roi_ind = "` to print `jpp.b_num - 1` to correctly describe the valid upper bound.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #506.
