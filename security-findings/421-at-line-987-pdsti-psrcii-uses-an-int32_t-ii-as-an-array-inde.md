# Security finding #421: At line 987, `pdst[i] = psrc[ii]` uses an `int32_t ii` as an array …

**Summary:** At line 987, `pdst[i] = psrc[ii]` uses an `int32_t ii` as an array …

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-129: Improper Validation of Array Index
**Severity / Impact:** An attacker supplying a crafted ONNX/IR model with a Gather node having a 1-D int32 data tensor (≤64 elements) can cause an out-of-bounds read of 4 bytes per crafted index. This can leak adjacent heap/stack memory (information disclosure) and may crash the process if the out-of-range address is unmapped (DoS). The 1DCase fast path is taken during both static and dynamic inference (execute() line 466, executeDynamicImpl() line 534), so any inference call with a matching model is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-supplied GATHER_INDICES tensor values (int32_t) cross into exec1DCase() at execute()/executeDynamicImpl() without any range gate on the index values themselves.

## Description / Root cause
At line 987, `pdst[i] = psrc[ii]` uses an `int32_t ii` as an array index with no post-adjustment bounds check. Two paths lead to OOB: (1) if `ii >= axisDim` (positive out-of-range), the `if (ii < 0)` guard at line 980 does not fire, so `ii` is used directly, reading past the end of the data buffer; (2) if `reverseIndexing` is true and the original index is below `-axisDim` (e.g., `-axisDim - 1`), the adjustment `ii += axisDim` at line 982 uses usual arithmetic conversions (int32_t sign-extended to uint64_t, added to size_t axisDim, truncated back to int32_t), yielding `ii = -1`, and `psrc[-1]` reads memory before the buffer. The `canOptimize1DCase` gate in `prepareParams()` (lines 395-403) only checks shape/rank/precision and that dimensions are ≤ 64 — it never validates that index VALUES are within `[−axisDim, axisDim)`. By contrast, `execReference()` (line 945-947) correctly casts `const size_t idx = ii` and guards with `if (idx < static_cast<size_t>(axisDim))` before any access.

**Validator analysis:** Confirmed real and reachable. exec1DCase (gather.cpp:967-989) is selected when dataSrcRank<=1, precision i32, and dims<=64 (prepareParams lines 396-403) — canOptimize1DCase inspects only shape/rank/precision, never index VALUES. execute()/executeDynamicImpl() (lines 466,534) dispatch to it for both static and dynamic inference. At line 987 `psrc[ii]` is indexed with no `ii < axisDim` gate: (1) a positive model-supplied index >= axisDim is used directly (OOB read past buffer); (2) reverseIndexing with index < -axisDim wraps via `ii += axisDim` where int32 ii is promoted to size_t, added, and truncated back to int32 = -1, giving psrc[-1] (OOB read before buffer). The safe execReference (lines 945-947) does exactly the missing check: `const size_t idx = ii; if (idx < axisDim)`. vulnType CWE-125/CWE-129 is accurate; impact (4-byte OOB read/info leak or DoS, since dtype is uint32) is accurate. The proposed fix is correct and sufficient — adding `const size_t idx = static_cast<size_t>(ii); if (idx >= axisDim) { pdst[i] = 0; continue; } pdst[i] = psrc[idx];` mirrors execReference and catches both under- and over-range cases (note: write 0 like execReference's memset, matching its semantics).

## Exploit / Proof of Concept
Craft an ONNX model with a Gather node where GATHER_DATA is a 1-D int32 tensor of length N (≤64) and GATHER_INDICES contains the value `-(N+1)`. After `ii += axisDim` (= `-(N+1) + N` = `-1`), `psrc[-1]` reads 4 bytes immediately before the data buffer. Alternatively, supply an index `≥ N` (e.g., `N+5`) to read 4 bytes beyond the buffer. Both paths pass the `canOptimize1DCase` check unchanged because only shape/precision, not index values, are inspected there.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in Gather::exec1DCase (gather.cpp:987).
// Pre-fix: with a 1-D i32 data tensor of length N (<=64) and an index value
//   either >= N or < -N (reverseIndexing), psrc[ii] reads out of bounds
//   (caught by ASan). Post-fix: out-of-range indices yield 0 in the output
//   (mirroring Gather::execReference at gather.cpp:945-947).
//
// Harness: ov_cpu_unit_tests (intel_cpu component).
// TODO: confirm exact include paths / helper names against the existing
//       gather single-layer tests under
//       openvino/src/plugins/intel_cpu/tests/unit (or functional/single_layer_tests/gather.cpp).

#include <gtest/gtest.h>
// TODO: include the intel_cpu test scaffolding used by the existing Gather
//       node unit tests (e.g. the node test fixture header). Names below are
//       placeholders until verified by reading the surrounding test tree.

namespace ov {
namespace intel_cpu {
namespace node_test {

// TODO: replace with the project's actual Gather-node unit-test fixture.
TEST(GatherNode1DCase, OutOfRangeIndexDoesNotReadOOB) {
    // 1-D int32 data of length N=4 (<=64) -> triggers canOptimize1DCase
    // (prepareParams gather.cpp:399-401).
    const std::vector<int32_t> data = {10, 20, 30, 40};
    const int32_t axisDim = static_cast<int32_t>(data.size());

    // Indices that exercise both OOB paths:
    //   N+5  -> positive over-range (psrc[ii] past buffer)
    //   -(N+1) -> reverseIndexing under-range -> ii==-1 -> psrc[-1]
    const std::vector<int32_t> indices = {0, axisDim + 5, -(axisDim + 1)};

    std::vector<uint32_t> out(indices.size());

    // TODO: build and run the Gather node 1-D path with reverseIndexing=true
    //       using the intel_cpu node test fixture instead of this direct call.
    //       runGather1DCase(data, indices, /*reverseIndexing=*/true, out);

    // Post-fix expectation: in-range index returns the element; every
    // out-of-range index returns 0 (execReference semantics). Pre-fix this
    // test instead trips ASan (heap-buffer-overflow READ of 4 bytes).
    EXPECT_EQ(out[0], 10u);
    EXPECT_EQ(out[1], 0u);
    EXPECT_EQ(out[2], 0u);
}

}  // namespace node_test
}  // namespace intel_cpu
}  // namespace ov
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter='GatherNode1DCase.OutOfRangeIndexDoesNotReadOOB' under AddressSanitizer. Pre-fix expected failure: ASan 'heap-buffer-overflow READ of size 4' inside Gather::exec1DCase (gather.cpp:987) for the over-range and the -1 (reverseIndexing) indices. Post-fix: test passes with out-of-range indices producing 0. TODO: wire the placeholder runGather1DCase helper to the real intel_cpu Gather node test fixture before use.

## Suggested fix
After the negative-index adjustment block (after line 986), add the same guard used in execReference(): `const size_t idx = static_cast<size_t>(ii); if (idx >= axisDim) { pdst[i] = 0; continue; }` then use `psrc[idx]` in place of `psrc[ii]`. This mirrors the existing validated path in execReference() (lines 945-962) and ensures both under-range (post-adjustment negative) and over-range indices are caught before the array access.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #421.
