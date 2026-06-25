# Security finding #426: Identical missing validation pattern as the ReduceNone overload: at…

**Summary:** Identical missing validation pattern as the ReduceNone overload: at…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound / CWE-787: Out-of-bounds Write
**Severity / Impact:** Same heap out-of-bounds write impact as the ReduceNone overload, but additionally triggered by SUM/PROD/MIN/MAX/SUB reduction modes, widening the attack surface to any ScatterNDUpdate operation with a non-NONE reduction — including the newly-added v15 SUB reduction exposed in the dispatch table at lines 1023-1033.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1107` — `ScatterUpdate::scatterNDUpdate<DataType,KernelType> (reduction overloads)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Same trust boundary: attacker-controlled indices tensor in model input → getIndicesValue() at line 1108 returns int64_t with no prior per-component validation

## Description / Root cause
Identical missing validation pattern as the ReduceNone overload: at line 1109-1112, negative-index correction is applied but `idxValue` is not subsequently checked to be in `[0, srcDataDim[i])`. At line 1113 — `dstOffset += idxValue * srcBlockND[i + 1]` — the same signed-to-unsigned implicit conversion and wrapping applies. The post-loop guard at line 1117 (`CPU_NODE_ASSERT(dstOffset < elementsCount)`) can be bypassed identically. At line 1121, `DataType* dstDataWithOffset = dstData + dstOffset` performs pointer arithmetic with the bypassed offset, and line 1124 (`kernel(dstDataWithOffset + idx, ...)`) writes through a reduction kernel to that location, potentially outside the valid allocation when pointer arithmetic wraps.

**Validator analysis:** The cited reduction overload (template body, lines 1083-1127) is genuinely missing per-component index validation, identical to the ReduceNone overload (1063-1079, source_work_item 352). At 1108 getIndicesValue returns attacker-controlled int64_t; the negative correction at 1109-1111 does not re-check that the corrected value is in range, so a residually-negative or huge index makes idxValue*srcBlockND[i+1] (signed→unsigned, 64-bit) wrap. The post-loop guard CPU_NODE_ASSERT(dstOffset<elementsCount) at 1117 only validates the slice base, not base+sizeToUpdate, and is itself bypassable by wraparound; the kernel at 1124 then writes srcBlockND[k] elements starting at dstData+dstOffset, allowing OOB heap writes. Reachable for any v15 ScatterNDUpdate with reduction != NONE via the OV_SWITCH dispatch at 1022-1032. The vulnType (CWE-190 → CWE-787) and impact (controlled heap OOB write) are accurate. The proposedFix is correct in direction but incomplete: a per-component check `if (idxValue<0 || (size_t)idxValue>=srcDataDim[i]) CPU_NODE_THROW(...)` after line 1111 closes the wrap and out-of-range path, but the guard at 1117 should additionally check `dstOffset + sizeToUpdate <= elementsCount` (not just dstOffset) since the write spans sizeToUpdate elements; the multiplication should also be done in a checked/wider form. The same hardening must be applied to the ReduceNone overload. Both fixes belong in the template body so all DataType/KernelType instantiations are covered.

## Exploit / Proof of Concept
Craft a v15::ScatterNDUpdate ONNX node with reduction=SUM. Supply an indices tensor with k=2 and multi-dimensional values crafted (as described above) so the accumulated dstOffset wraps to a value < elementsCount. The SUM kernel at line 1124 then reads/writes `dstData + dstOffset + idx` for idx in [0, sizeToUpdate), with sizeToUpdate = srcBlockND[k]. An attacker controlling both the index tuple and the `updates` tensor controls both the write destination (within the wrapped range) and the written value, enabling arbitrary heap writes.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1107-1124
// (reduction overload of ScatterUpdate::scatterNDUpdate). Pre-fix: an out-of-range / negative
// indices tuple makes dstOffset wrap past the CPU_NODE_ASSERT at line 1117 and the reduction
// kernel at line 1124 writes OOB on the heap (ASan heap-buffer-overflow). Post-fix: the per-
// component range check throws, so the op is rejected before any write.
//
// TODO: target = ov_cpu_unit_tests (intel_cpu/tests/unit). Confirm exact fixture/helpers by
//       reading intel_cpu/tests/unit for an existing ScatterNDUpdate node single-layer test;
//       symbol names below are placeholders and MUST be replaced with the real harness API.
#include <gtest/gtest.h>
// TODO: include the real intel_cpu unit-test node-builder / ov::Model + ov::Core helpers.

TEST(scatter_nd_update_cpu, reduction_sum_rejects_out_of_range_indices) {
    // TODO: build an ov::Model with a v15 ScatterNDUpdate(reduction=SUM):
    //   data    : f32 shape {4}            (elementsCount = 4)
    //   indices : i64 shape {1,1}  value = {-1000000}  // residually negative -> wraps dstOffset
    //   updates : f32 shape {1}
    // TODO: compile on the CPU plugin and infer.
    // Expectation AFTER fix: per-component validation throws.
    // EXPECT_THROW(compiledModel.create_infer_request().infer(), ov::Exception);
    //
    // Pre-fix behaviour: no throw; ASan reports heap-buffer-overflow WRITE inside the
    // reduction kernel (scatter_update.cpp:1124).
    GTEST_SKIP() << "TODO: wire up ov_cpu_unit_tests ScatterNDUpdate fixture";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=scatter_nd_update_cpu.reduction_sum_rejects_out_of_range_indices . Expected pre-fix (ASan build): heap-buffer-overflow WRITE in ScatterUpdate::scatterNDUpdate reduction kernel (scatter_update.cpp:1124); post-fix: ov::Exception thrown from per-component index validation, test passes.

## Suggested fix
Apply the same fix as for the ReduceNone overload: add `if (idxValue < 0 || static_cast<size_t>(idxValue) >= srcDataDim[i]) { CPU_NODE_THROW(...); }` immediately after line 1111 (`idxValue += srcDataDim[i]`) and before line 1113. Perform the multiplication with overflow detection. This fix must be applied to ALL instantiations of the templated overload (i.e., it should be in the template body, not an individual specialization).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #426.
