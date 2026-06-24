# Security finding #189: Line 987 (`pdst[i] = psrc[ii]`) performs the gather read with no bo…

**Summary:** Line 987 (`pdst[i] = psrc[ii]`) performs the gather read with no bo…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Heap out-of-bounds read from the source data buffer (up to `axisDim` elements before the buffer start, or one element past its end). Can leak sensitive heap metadata or adjacent allocation contents, or trigger a segfault/crash (DoS). Exploitable whenever an ONNX model or other caller supplies a GATHER node with a 1-D source tensor (≤64 elements) and out-of-range indices.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** GATHER_INDICES input tensor: attacker-controlled int32_t index values arrive from the model/user input at runtime and are read directly via pidx[i] (line 979).

## Description / Root cause
Line 987 (`pdst[i] = psrc[ii]`) performs the gather read with no bounds check after the normalization block (lines 980–986). `execReference()` protects the equivalent read at line 954 with `if (idx < static_cast<size_t>(axisDim))` (line 947), but `exec1DCase()` has no such guard. Two OOB paths exist: (1) reverseIndexing=false, any negative index → `ii` is set to `axisDim` (line 984), i.e. one element past the end of the source array; (2) reverseIndexing=true, index < -axisDim → after `ii += axisDim` (line 982) `ii` is still negative, and using a negative int32_t as the subscript into `psrc` (a uint32_t*) reads memory before the allocation. The `canOptimize1DCase` gate (lines 396–403) only enforces rank≤1 and element-count≤64; it never validates index values.

**Validator analysis:** Confirmed real and reachable. exec1DCase (gather.cpp:967-989) normalizes only negative indices: reverseIndexing=false sets ii=axisDim (one-past-end), reverseIndexing=true with idx<-axisDim leaves ii negative (read before buffer). Critically, positive indices >= axisDim are NEVER checked either, so psrc[ii] at line 987 reads OOB. execReference at line 947 has the `if (idx < axisDim)` guard that this fast path omits. canOptimize1DCase (lines 396-403) gates only on rank/precision/element-count (≤64), never on index values, and indices are runtime attacker-controlled (pidx[i], line 979). CWE-125 OOB read and the heap-leak/DoS impact are accurate. The proposed fix `if (static_cast<size_t>(ii) >= axisDim) { pdst[i]=0; continue; }` placed after the normalization block is correct and sufficient: the size_t cast makes any residual-negative ii wrap to a huge value that fails `< axisDim`, and it also catches the positive-overflow case (idx>=axisDim) the report under-emphasized; the zero-fill matches execReference's else-branch memset semantics.

## Exploit / Proof of Concept
Construct a model with a Gather node, axis=0, source tensor of shape [5] (dtype int32), and GATHER_INDICES input = [-10]. With reverseIndexing=true: ii=-10, ii += 5 → ii=-5 (still negative). Line 987 executes `psrc[-5]`, reading 5 uint32_t words (20 bytes) before the heap allocation, leaking heap data or crashing. With reverseIndexing=false and any negative index: ii is set to 5 (axisDim), and `psrc[5]` is the one-past-end element — also OOB.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for OOB read at intel_cpu/src/nodes/gather.cpp:987 (Gather::exec1DCase).
// Pre-fix: ASan reports heap-buffer-overflow READ when a 1-D i32 source (<=64 elems)
// is gathered with an out-of-range index (e.g. -10 with reverseIndexing, or 100).
// Post-fix (guard `if (static_cast<size_t>(ii) >= axisDim) { pdst[i]=0; continue; }`):
// out-of-range lanes are zero-filled and no OOB occurs.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). The 1-D fast path requires
// dataSrcRank<=1, i32 precision, dataDims[0]<=64, idxDims[0]<=64 (gather.cpp:396-403).
#include <gtest/gtest.h>
// TODO: include the intel_cpu node test fixtures used by intel_cpu/tests/unit
//       (e.g. the Graph/Node test helpers) — confirm exact headers/targets by reading
//       intel_cpu/tests/unit/CMakeLists.txt and an existing node unit test.

// TODO: Build a minimal CPU graph containing a single Gather node:
//   - GATHER_DATA: shape [5], precision i32, values {0,1,2,3,4}
//   - GATHER_INDICES: shape [1], precision i32, value {-10} (reverseIndexing=true case)
//   - axis = 0
// so prepareParams() sets canOptimize1DCase=true and exec1DCase() runs.
TEST(GatherExec1DCase, OutOfRangeIndexIsBoundsChecked) {
    // TODO: construct node, allocate src/idx/dst memory via the unit-test memory helpers,
    //       set pidx[0] = -10, run execute(), and assert dst[0] == 0 (zero-fill semantics).
    //       The decisive signal pre-fix is the ASan heap-buffer-overflow on psrc[ii].
    GTEST_SKIP() << "TODO: wire up intel_cpu node fixture to drive Gather::exec1DCase";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests; Run: ov_cpu_unit_tests --gtest_filter=GatherExec1DCase.* ; Expected pre-fix under ASan: 'heap-buffer-overflow READ' in Gather::exec1DCase at gather.cpp:987 (read of psrc[ii]); post-fix the test passes with dst[0]==0 and no ASan report.

## Suggested fix
Mirror the guard from `execReference()`. After the normalization block and before line 987, add: `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` This rejects any index that remains negative (wraps to a large size_t, fails `< axisDim`) as well as any non-negative out-of-range index, matching the exact semantic of the reference path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #189.
