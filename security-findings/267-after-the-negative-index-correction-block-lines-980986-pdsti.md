# Security finding #267: After the negative-index correction block (lines 980–986), `pdst[i]…

**Summary:** After the negative-index correction block (lines 980–986), `pdst[i]…

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-129: Improper Validation of Array Index
**Severity / Impact:** An attacker-supplied GATHER_INDICES tensor can read arbitrary heap memory before or after the data buffer, enabling an information-leak of process heap contents (models, weights, other tensor data) or a crash/DoS. The `canOptimize1DCase` path is enabled when `dataSrcRank <= 1`, precision is `i32`, and both data and index dimensions are ≤ 64 — a configuration commonly hit in shape-inference subgraphs, making the reach condition very realistic.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvino
**Trust boundary:** GATHER_INDICES input port — runtime-supplied int32_t array, fully attacker-controlled during inference

## Description / Root cause
After the negative-index correction block (lines 980–986), `pdst[i] = psrc[ii]` (line 987) is executed with no bounds guard. Three distinct OOB sub-cases exist: (A) positive `ii >= axisDim` — `psrc[ii]` reads past the end of the data buffer; (B) when `reverseIndexing == false` and `ii < 0`, `ii` is set to `axisDim` (line 984), then `psrc[axisDim]` reads exactly one element past the end (off-by-one); (C) when `reverseIndexing == true` and `ii` is sufficiently negative (e.g., ii=-100, axisDim=5), `ii += axisDim` leaves ii still negative (ii=-95), and `psrc[-95]` reads arbitrarily before the start of the buffer. In contrast, `execReference()` at line 947 wraps the same read in `if (idx < static_cast<size_t>(axisDim))`, which also catches remaining-negative values via unsigned wraparound. `exec1DCase()` has no such guard whatsoever.

**Validator analysis:** Confirmed real and reachable. exec1DCase() (gather.cpp:967-989) reads the untrusted index pidx[i] (GATHER_INDICES, runtime tensor) into ii and after the negative-correction block at 980-986 dereferences psrc[ii] at line 987 with no upper-bound check. All three sub-cases hold: (A) positive ii>=axisDim reads past the data buffer; (B) reverseIndexing==false sets ii=axisDim (line 984), an off-by-one read at psrc[axisDim]; (C) reverseIndexing==true with a strongly negative index leaves ii negative after ii+=axisDim (line 982), reading before the buffer. execReference's parallel body (gather.cpp:937-947) demonstrates the intended guard `if (idx < static_cast<size_t>(axisDim))`, absent here. The path is reached directly from execute()/executeDynamicImpl() (lines 466-468, 534-536) when prepareParams enables canOptimize1DCase for 1-D i32 data with dims<=64 (lines 396-403) — a realistic shape-infer subgraph config. vulnType (CWE-125/CWE-129) and impact (heap info-leak / OOB read DoS) are accurate. The proposed fix is correct and sufficient: convert ii to size_t and reject idx>=axisDim (writing 0 like execReference), which also catches the remaining-negative reverseIndexing case via unsigned wraparound; sharing a helper with execReference is the cleaner consolidation. Note the fix should also zero pdst[i] for the rejected case to match execReference semantics rather than leaving it uninitialized.

## Exploit / Proof of Concept
Submit a model with a 1-D int32 GATHER_DATA tensor (e.g., 5 elements) and a GATHER_INDICES tensor containing the value 65 (positive OOB), -1 with reverseIndexing=false (fence-post, sets ii=5=axisDim), or -100 with reverseIndexing=true (ii=-95). `prepareParams()` sets `canOptimize1DCase=true` (lines 396–403) because dataSrcRank==1 and dims ≤ 64. At inference time, `execute()` or `executeDynamicImpl()` calls `exec1DCase()` directly (lines 466–468, 534–536) with no interposing validation. `pidx[0]` is read as the untrusted index, no bounds check exists, and `psrc[ii]` reads out of bounds.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for OOB read at openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
// (Gather::exec1DCase reads psrc[ii] with no bounds check on the runtime-supplied
//  GATHER_INDICES value). The assertion: with a 1-D i32 data tensor of 5 elements and
//  an out-of-range index (65, or -1 with reverseIndexing=false, or a strongly negative
//  index with reverseIndexing=true), inference must NOT read out of bounds — under ASan
//  the pre-fix code traps; the post-fix code returns 0 for the offending position.
//
// TODO: This is a SKELETON. The CPU plugin unit harness (ov_cpu_unit_tests) builds Gather
//   subgraph tests via ov::test::SubgraphBaseTest / ngraph ops. Fill in the exact include
//   paths and helper names by reading src/plugins/intel_cpu/tests/unit/ before use.
#include <gtest/gtest.h>
// TODO: #include the CPU node test fixtures, e.g. "nodes/..." and openvino/op/gather.hpp

TEST(GatherExec1DCaseOOB, RejectOutOfRangeIndex) {
    // TODO: build a Gather op:
    //   data  = Constant<i32>{shape={5}, values={10,11,12,13,14}}  (dataSrcRank==1, dims<=64 -> canOptimize1DCase)
    //   axis  = 0
    //   indices = Parameter<i32>{shape={1}}  // attacker-controlled GATHER_INDICES
    // Compile for CPU, then run inference three times with indices = {65}, {-1}, {-100}.
    //
    // Pre-fix: exec1DCase() dereferences psrc[ii] OOB -> ASan heap-buffer-overflow.
    // Post-fix: out-of-range indices produce 0 in the output (matching execReference()).
    //
    // EXPECT_NO_THROW(infer_with_index(65));   // and no ASan abort
    // EXPECT_EQ(out_value, 0);
    GTEST_SKIP() << "TODO: wire up CPU Gather subgraph fixture (see intel_cpu/tests/unit/)";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter='GatherExec1DCaseOOB.*'. With AddressSanitizer enabled, the pre-fix build aborts with 'heap-buffer-overflow READ' inside Gather::exec1DCase (gather.cpp:987); the post-fix build passes and yields 0 for out-of-range indices.

## Suggested fix
Apply the same idiom used in `execReference()`: after the negative-index correction block and before the read, add `const size_t idx = static_cast<size_t>(ii); if (idx >= static_cast<size_t>(axisDim)) { pdst[i] = 0; continue; }` and use `psrc[idx]` instead of `psrc[ii]`. This converts `ii` to unsigned (catching remaining-negative values via wraparound) and rejects any out-of-range index. Alternatively, consolidate into a helper that is shared with `execReference()`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #267.
