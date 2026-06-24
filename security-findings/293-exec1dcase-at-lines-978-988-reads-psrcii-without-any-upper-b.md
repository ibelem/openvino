# Security finding #293: exec1DCase() at lines 978-988 reads psrc[ii] without any upper-boun…

**Summary:** exec1DCase() at lines 978-988 reads psrc[ii] without any upper-boun…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read from the source data buffer. An attacker supplying a crafted ONNX model can leak arbitrary process memory (e.g., heap metadata, adjacent tensor data) into the output tensor, or trigger a crash/DoS. This affects any OpenVINO deployment that loads untrusted models.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:978` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model index tensor (GATHER_INDICES) forwarded by the OpenVINO EP to Gather::exec1DCase

## Description / Root cause
exec1DCase() at lines 978-988 reads psrc[ii] without any upper-bound check on ii. Two cases enable OOB access: (1) when ii < 0 and reverseIndexing is false, ii is assigned exactly axisDim (line 984), making psrc[axisDim] a one-past-the-end OOB read; (2) when ii >= 0 but ii >= axisDim (a positive out-of-range index), no check guards the access. In contrast, execReference() at line 947 correctly gates the read with 'if (idx < static_cast<size_t>(axisDim))'—exec1DCase has no equivalent guard.

**Validator analysis:** Confirmed real OOB read. exec1DCase (gather.cpp:967-989) clamps negative indices to axisDim (one-past-end) or leaves positive out-of-range indices unchecked, then dereferences psrc[ii] (uint32_t*) at line 987 with no `ii < axisDim` gate. The parallel execReference path (lines 938-963) deliberately zeroes out-of-range indices (guard at line 947), proving the intended/safe semantics and that exec1DCase regresses them. Reachability is solid: prepareParams() sets canOptimize1DCase when dataSrcRank<=1, i32 precision, and both data/idx dims<=64 (gather.cpp:396-403), and both execute() (466-468) and executeDynamicImpl() (534-536) dispatch unconditionally to exec1DCase under that flag. CWE-125 Out-of-bounds Read is accurate; impact (info leak of adjacent heap into output / crash DoS from untrusted model) is correct, though the read is bounded to a uint32 element so it's a modest leak per element, repeatable across idxCnt (<=64) elements. The proposed fix is correct and sufficient: mirror execReference by bounds-checking ii in [0,axisDim) and writing 0 otherwise. One refinement — keep ii as a signed/wide type during the add (`ii += (int32_t)axisDim`) to avoid signed/unsigned mix, exactly as shown in the proposed loop body.

## Exploit / Proof of Concept
Craft an ONNX model whose Gather node has: (a) a 1-D int32 data input of size ≤ 64 (e.g., [1,2,3]), (b) a 1-D indices input of size ≤ 64 containing an out-of-range value (e.g., [99]). prepareParams() sets canOptimize1DCase=true and returns early (lines 395-403). execute()/executeDynamicImpl() then call exec1DCase(). With axisDim=3 and pidx[0]=99, the code reaches line 987 with ii=99 and executes psrc[99], reading 99*4=396 bytes past the start of a 3-element buffer. Alternatively, pidx[0]=-5 with reverseIndexing=false sets ii=axisDim=3, reading psrc[3] (one-past-the-end).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for OOB read at intel_cpu/src/nodes/gather.cpp:987 (exec1DCase).
// Pre-fix: psrc[ii] is read with ii==axisDim (negative idx, reverseIndexing=false)
// or ii>=axisDim (positive out-of-range idx) -> ASan heap-buffer-overflow read.
// Post-fix: out-of-range indices must yield 0 in the output, matching execReference
// (gather.cpp:947). Built for ov_cpu_unit_tests (subgraph/node test style).
//
// TODO(skeleton): confirm exact include paths + the intel_cpu node-test fixture
// helper (e.g. the GraphContext/Node build helpers under
// src/plugins/intel_cpu/tests/unit/). The 1D fast path requires:
//   - GATHER_DATA: 1-D i32 tensor, dim0 <= 64 (e.g. {1,2,3})
//   - GATHER_INDICES: 1-D i32 tensor, dim0 <= 64 with an OOB value (e.g. {99} or {-5})
//   - reverseIndexing == false on the Gather op
// so that prepareParams() sets canOptimize1DCase=true (gather.cpp:399-401) and
// exec1DCase() runs.
#include <gtest/gtest.h>
// TODO: include the intel_cpu node/graph unit-test harness headers used by the
// existing tests under src/plugins/intel_cpu/tests/unit/ (graph builder, infer).

TEST(GatherExec1DCase, OutOfRangeIndexIsZeroedNotOOBRead) {
    // TODO: build a Gather op with 1-D i32 data {1,2,3} (axisDim=3),
    //       1-D i32 indices {99} (positive OOB) and reverseIndexing=false.
    // TODO: run inference on the CPU plugin so Gather::exec1DCase executes.
    // Expected post-fix: output[0] == 0 (no OOB read). Pre-fix: ASan flags
    // heap-buffer-overflow at gather.cpp:987.
    // ASSERT_EQ(out[0], 0u);
    GTEST_SKIP() << "skeleton: wire up intel_cpu Gather node fixture";
}

TEST(GatherExec1DCase, NegativeIndexNoReverseIsZeroedNotOnePastEnd) {
    // TODO: data {1,2,3} (axisDim=3), indices {-5}, reverseIndexing=false.
    // Pre-fix: ii is set to axisDim=3 (gather.cpp:984) -> psrc[3] one-past-end read.
    // Expected post-fix: output[0] == 0.
    GTEST_SKIP() << "skeleton: wire up intel_cpu Gather node fixture";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter='GatherExec1DCase.*' under ASan. Expected pre-fix: AddressSanitizer heap-buffer-overflow READ of size 4 at intel_cpu/src/nodes/gather.cpp:987 (psrc[ii] with ii>=axisDim). Post-fix: tests pass with out[0]==0.

## Suggested fix
Mirror the guard from execReference(): after adjusting ii for negative indices, add an explicit bounds check before the array access. Replace the loop body with:
  if (ii < 0) { if (reverseIndexing) ii += (int32_t)axisDim; else ii = (int32_t)axisDim; }
  if (ii >= 0 && (size_t)ii < axisDim) { pdst[i] = psrc[ii]; } else { pdst[i] = 0; }
This ensures that indices outside [0, axisDim) produce a zeroed output rather than an OOB read, consistent with the behavior of execReference().


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #293.
