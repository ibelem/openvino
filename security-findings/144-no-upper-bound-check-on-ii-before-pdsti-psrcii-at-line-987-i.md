# Security finding #144: No upper-bound check on `ii` before `pdst[i] = psrc[ii]` at line 98…

**Summary:** No upper-bound check on `ii` before `pdst[i] = psrc[ii]` at line 98…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds heap read: up to `(INT32_MAX - axisDim) * 4` bytes past the source data buffer can be read and written to the destination tensor output, leaking adjacent heap contents (weights, other model data, runtime metadata) to any consumer of the inference output. Since `canOptimize1DCase` is set for int32 tensors up to size 64 (the common shape-inference subgraph path), this is reachable with a small crafted ONNX model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled ONNX Gather indices tensor flowing from the OpenVINO EP plugin layer into the CPU-node execution path

## Description / Root cause
No upper-bound check on `ii` before `pdst[i] = psrc[ii]` at line 987. `ii` is read as `int32_t` from the model-supplied indices tensor (`pidx[i]`, line 979), the only guard is the `ii < 0` branch (lines 980-985). If `ii >= axisDim` (a positive out-of-range index), `psrc[ii]` reads past the end of the data tensor's allocation. `execReference()` performs the equivalent check at line 947 (`if (idx < static_cast<size_t>(axisDim))`), but `exec1DCase()` omits it entirely.

**Validator analysis:** vulnType (CWE-125 OOB read) is accurate: a positive in-range-of-int32 but >=axisDim index produces psrc[ii] past the data allocation, copied into the output — heap disclosure. Note an additional sub-bug the report under-states: the non-reverse negative branch sets ii=axisDim (line 984), itself one-past-end and unguarded. Impact (leaking adjacent heap into output) is correct; the 'up to (INT32_MAX-axisDim)*4 bytes' figure is theoretical (bounded by page faults), but a real OOB read remains. Reachability is genuine because canOptimize1DCase is selected purely on precision/shape (i32, dims<=64), so an explicit attacker Gather op with a runtime data input (preventing constant-folding) plus a constant out-of-range index hits exec1DCase at inference. proposedFix is correct and sufficient: inserting `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` after the negative block mirrors execReference's zero-fill, and because the cast to size_t turns any residual negative ii (e.g. reverseIndexing under-correction) into a huge value it is also caught — so the single guard fixes both the positive-OOB and the ii=axisDim/negative-residual cases.

## Exploit / Proof of Concept
Craft an ONNX model with a 1-D int32 Gather data tensor of shape [4] and a 1-D int32 indices tensor [1] containing value 0x7FFFFFFF. OpenVINO EP accepts the model, compiles it, and during inference `exec1DCase()` is dispatched (dataSrcRank=1, precision=i32, dims≤64). The loop reads `pidx[0] = 0x7FFFFFFF`, `ii` is positive so the `< 0` branch is skipped, and `psrc[0x7FFFFFFF]` reads ~8 GB past the buffer start. In practice the effective forward distance is bounded by OS page faults, but any readable heap page in range will have its 4-byte word copied to the output tensor and returned to the caller.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read in Gather::exec1DCase()
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no upper-bound check on `ii` (only ii<0 handled, 980-986).
// Pre-fix: with a positive out-of-range index this reads past the i32 data buffer
//   (ASan: heap-buffer-overflow READ of size 4 in ov::intel_cpu::node::Gather::exec1DCase).
// Post-fix: index >= axisDim must be zero-filled (mirroring execReference() line 947/959-963),
//   so the output element for the OOB index is 0 and no OOB access occurs.
//
// Target harness: ov_cpu_unit_tests (gtest + ASan).
// NOTE: emitted as a SKELETON. The intel_cpu unit-test tree builds Gather via the
// internal graph/Node infra (see src/plugins/intel_cpu/tests/unit/ for the exact
// helpers/fixtures). Exact symbol names for constructing a single CPU Gather node
// with const data + const indices and invoking execute() were not verified read-only,
// so the construction steps below are TODOs.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

// TODO: include the intel_cpu test helpers actually used under
//       src/plugins/intel_cpu/tests/unit/ (graph builder / node fixtures).

TEST(GatherCpuNode1DCase, OutOfRangeIndexIsZeroFilledNotOOBRead) {
    // Data: 1-D i32 tensor, axisDim = 4  (<=64 so canOptimize1DCase path is selected,
    //       see gather.cpp:396-404). Keep it as a runtime input so Gather is NOT
    //       constant-folded and the CPU node actually executes.
    const std::vector<int32_t> data = {10, 11, 12, 13};
    // Indices: 1-D i32 tensor with a positive out-of-range value.
    const std::vector<int32_t> indices = {0x7FFFFFFF};

    // TODO: build a single CPU Gather node (axis=0, batchDims=0, reverseIndexing as default)
    //       with `data` on GATHER_DATA and `indices` on GATHER_INDICES, allocate the
    //       output, run prepareParams() + execute(). Use the intel_cpu unit-test graph
    //       fixture rather than the full plugin so exec1DCase() is exercised directly.
    // std::vector<int32_t> out(indices.size());
    // runGatherNode(data, {4}, indices, {1}, /*axis*/0, out);

    // Post-fix expectation: out-of-range index -> zero-fill (consistent with execReference).
    // EXPECT_EQ(out[0], 0);
    // Pre-fix: the line above never executes cleanly; ASan aborts on the OOB read at
    //          gather.cpp:987 before reaching here.
    GTEST_SKIP() << "TODO: wire up intel_cpu Gather node construction helpers from "
                    "src/plugins/intel_cpu/tests/unit/ to drive exec1DCase().";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter='GatherCpuNode1DCase.OutOfRangeIndexIsZeroFilledNotOOBRead'. Expected pre-fix failure: AddressSanitizer 'heap-buffer-overflow READ of size 4' inside ov::intel_cpu::node::Gather::exec1DCase (gather.cpp:987). Post-fix: the OOB index is zero-filled and the test passes.

## Suggested fix
After the negative-index adjustment block (after line 985), add an upper-bound guard mirroring `execReference()`: `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }`. This clamps out-of-range indices to a zero-fill (consistent with the existing out-of-bounds behavior in `execReference()` lines 959-963).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #144.
