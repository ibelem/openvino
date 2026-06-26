# Security finding #496: Lines 427–429 repeat the same unguarded chained multiplications as …

**Summary:** Lines 427–429 repeat the same unguarded chained multiplications as …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Same as the static-shape path: overflowed `srcAfterBatchSizeInBytes` is passed to the JIT kernel at execute:494 as a pointer stride, enabling out-of-bounds memory access (OOB read/write) on every inference call with a malicious shape, potentially crashing the process or enabling code execution.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:427` — `Gather::prepareParams()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-supplied tensor shape dimensions from untrusted ONNX/IR model graph at inference time (dynamic shape path)

## Description / Root cause
Lines 427–429 repeat the same unguarded chained multiplications as lines 173–175 but on the dynamic-shape path (when `!isDataShapeStat || !isAxisInputConst`). `axisDim * afterAxisSizeInBytes` at line 427 (axisDim is `int`, afterAxisSizeInBytes is `uint64_t`) and `betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes` at line 429 (both uint64_t) have no overflow guard. These execute at every inference step when shapes are dynamic, so an attacker who controls dynamic input shapes can trigger the overflow at runtime.

**Validator analysis:** The CWE-190 narrowing is real: axisDim is declared `int` (gather.h:92) but is assigned from `dataDims[axis]` (a size_t VectorDims element) at line 418 on the dynamic path, identical to the static path's line 164. When axisDim exceeds INT_MAX it truncates/sign-wraps; in the line-427 multiply `axisDim * afterAxisSizeInBytes` the (possibly negative) int promotes to a huge uint64_t, corrupting axisAndAfterAxisSizeInBytes and srcAfterBatchSizeInBytes, which become kernel strides at execute:493-494 -> OOB. No magnitude guard exists (the only check at line 411 validates axis index, not the dimension value). The line-429 claim of uint64_t*uint64_t overflow is NOT a genuine defect on its own (would require >2^64-byte tensors, impossible); the real and only exploitable root cause is the int axisDim narrowing surfacing at line 427. Reachability caveat: triggering requires a real allocated data tensor with an axis dimension >2^31 (>~2GB buffer), since getStaticDims() reflects actually-allocated memory — feasible but resource-heavy, not free per-inference as worded. vulnType (CWE-190) is accurate; impact (OOB via corrupt stride) is accurate but practical bar is high. The proposed fix is correct and the better/sufficient form: widen axisDim to int64_t (gather.h:92) so the size_t dim no longer truncates; the explicit OPENVINO_ASSERT overflow guards at 427/429 are good defense-in-depth but the line-429 uint64_t guard is largely cosmetic. Widening axisDim is the essential fix.

## Exploit / Proof of Concept
Serve a dynamic-shape Gather node over an inference API (e.g., ORT with OpenVINO EP). At inference time, supply an input tensor whose axis dimension is 2^32 (fits in uint64_t but > INT_MAX). `axisDim` at line 418 gets assigned the truncated `int` value (0 due to truncation mod 2^31 or a negative wrap). The multiplication at line 427 yields 0 or ~UINT64_MAX for `axisAndAfterAxisSizeInBytes`, and subsequently `srcAfterBatchSizeInBytes` is 0 or massive. The JIT kernel at execute:494 uses this zero/corrupt stride to index the source buffer, causing OOB memory access at each batch step.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 narrowing at gather.cpp:418/427 (axisDim is `int` in gather.h:92
// but assigned from size_t getStaticDims()[axis]). Pre-fix, an axis dimension > INT_MAX
// truncates/sign-wraps and corrupts srcAfterBatchSizeInBytes used as a JIT kernel stride
// (execute:493-494) -> OOB. Post-fix (axisDim widened to int64_t + overflow guard) the
// large dimension is represented faithfully / rejected and no OOB stride is produced.
//
// NOTE: a fully self-contained, compilable test is NOT achievable here: triggering the
// real overflow needs a Gather data tensor with an axis dimension > 2^31 elements
// (a multi-GB allocation), which a unit test cannot allocate. This is therefore a
// SKELETON. Exact target/symbols (ov_cpu_unit_tests, Gather node test fixtures) must be
// confirmed against the intel_cpu tests tree before use.

#include <gtest/gtest.h>
// TODO: include the intel_cpu Gather node header and the unit-test graph/infer fixtures
//       used by other ov_cpu_unit_tests (confirm exact paths under
//       openvino/src/plugins/intel_cpu/tests/unit/).

TEST(GatherIntegerOverflow, AxisDimNarrowingProducesNoOObStride) {
    // TODO: build a dynamic-shape Gather node (isDataShapeStat == false OR
    //       isAxisInputConst == false) so prepareParams takes the line 416-429 branch.
    // TODO: provide a data tensor whose axis static dim exceeds INT_MAX (e.g. (1LL<<31)).
    //       Because real allocation of >2GB is impractical, instead unit-test the arithmetic
    //       directly: feed dataDims[axis] = (1ULL<<31) into the computation and assert
    //       axisAndAfterAxisSizeInBytes / srcAfterBatchSizeInBytes equal the exact 64-bit
    //       product (no truncation/wrap), e.g.:
    //
    //       const uint64_t axisDim = (1ULL << 31);
    //       const uint64_t afterAxisSizeInBytes = 4; // f32, afterAxisSize==1
    //       const uint64_t expected = axisDim * afterAxisSizeInBytes; // 8589934592
    //       EXPECT_EQ(node->axisAndAfterAxisSizeInBytes(), expected);
    //
    // Pre-fix this fails because `int axisDim` truncates (1<<31) to INT_MIN, yielding a
    // wrong/huge uint64_t product; post-fix (int64_t axisDim) it matches `expected`.
    GTEST_SKIP() << "Skeleton: wire to ov_cpu_unit_tests Gather fixtures; see TODOs.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests . Run: ov_cpu_unit_tests --gtest_filter=GatherIntegerOverflow.* . Pre-fix the assertion on axisAndAfterAxisSizeInBytes/srcAfterBatchSizeInBytes mismatches (int axisDim truncation); with ASan and a true >2^31 axis dim the corrupted stride at gather.cpp execute:493-494 triggers a heap-buffer-overflow read/write. Post-fix (int64_t axisDim + overflow guard) the products are exact and no OOB occurs.

## Suggested fix
Widen `axisDim` to `int64_t` in gather.h:92 and add pre-multiplication overflow guards at lines 427 and 429 (matching the fix for lines 173/175). For example, before line 427: `OPENVINO_ASSERT(afterAxisSizeInBytes == 0 || static_cast<uint64_t>(axisDim) <= std::numeric_limits<uint64_t>::max() / afterAxisSizeInBytes, "Gather: axisAndAfterAxisSizeInBytes overflow");` and similarly for line 429.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #496.
