# Security finding #494: At lines 358–359, `p.dataBeforeAxisSumInBytes[j] = ((dstStart + j) …

**Summary:** At lines 358–359, `p.dataBeforeAxisSumInBytes[j] = ((dstStart + j) …

**CWE IDs:** CWE-197: Numeric Truncation Error → CWE-125: Out-of-bounds Read
**Severity / Impact:** The truncated `dataBeforeAxisSumInBytes` values are passed as `arg.dataBeforeAxisSumB = p.dataBeforeAxisSumInBytes.data()` at execute():500 and used by the JIT kernel as byte offsets from `arg.src` (the raw source data pointer). A wrong offset directs the kernel to read from attacker-controlled positions in the process address space, constituting an OOB read that can leak heap data or cause a crash/DoS.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:358` — `Gather::createPrimitive()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-supplied tensor dimensions → prepareParams → createPrimitive per-thread param initialisation → JIT kernel byte offsets

## Description / Root cause
At lines 358–359, `p.dataBeforeAxisSumInBytes[j] = ((dstStart + j) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;` computes a `uint64_t` result that is implicitly narrowed to `int` (because `dataBeforeAxisSumInBytes` is `std::vector<int>`, gather.h:43). `axisAndAfterAxisSizeInBytes` is a `uint64_t` field holding `axisDim * afterAxisSize * dataTypeSize`; for a model with large-dimensional data tensors this exceeds 2^31. The same overflow path also appears in `specIdxInBytes[j]` at line 354 and `idxBatchSumInBytes[j]` at lines 355–357 when `specIndicesSize * idxTypeSize` overflows 32 bits.

**Validator analysis:** The truncation is real: the threadExecParams byte-offset vectors are std::vector<int> (gather.h:39-43) while the values computed at gather.cpp:354-359 are uint64_t products (axisAndAfterAxisSizeInBytes = axisDim*afterAxisSize*dataTypeSize, and specIndicesSize*idxTypeSize). When the source data tensor exceeds ~2^31 bytes, the offset wraps to a wrong/negative int and is fed to the JIT kernel as a byte offset from arg.src (execute():500/498/499), enabling an OOB read — so CWE-197→CWE-125 is accurately categorised. Reachability caveat: the static JIT path (createPrimitive:337 !isDynamicNode, 304-307 afterAxisSize==1 or <=idxElPerVec) only triggers truncation once the gathered data buffer is physically >2GB, which requires the attacker to also supply/allocate a multi-GB tensor — practically heavy but within the model-dimension trust boundary, so the defect is genuine and reachable, not pre-mitigated (no size guard exists in prepareParams). The proposed fix is correct in direction but incomplete: widening the three vectors to int64_t alone is insufficient because the JIT kernel (jitUniGatherKernel) reads these arrays with 32-bit element strides; the kernel arg structs and gather access width must be updated in lockstep, or the values will still be truncated at the asm level. The short-term CPU_NODE_ASSERT gate on axisAndAfterAxisSizeInBytes (and on specIndicesSize*idxTypeSize) in prepareParams is the safer, sufficient mitigation and should be the primary fix.

## Exploit / Proof of Concept
Same trigger as Finding 1: provide a model tensor where `axisDim * afterAxisSize * dataTypeSize > 2^31`. The per-thread `dataBeforeAxisSumInBytes` computed in `createPrimitive:358-359` wraps upon truncation to `int`. Because `createPrimitive` is called once at graph-build time and the values are cached in `execParamsPerThread`, every subsequent `execute()` call reuses the corrupted offsets.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for gather.cpp:358-359 (and 354-357): byte offsets computed as
// uint64_t must not be truncated into std::vector<int> (gather.h:39-43). Pre-fix, a
// Gather whose data buffer exceeds INT32_MAX bytes yields wrapped/negative offsets and
// an OOB read in the JIT kernel (ASan heap-buffer-overflow). Post-fix, the node must
// either use 64-bit offsets or reject the oversized stride in prepareParams.
//
// NOTE: a real trigger needs a >2GB source tensor, which is not feasible to allocate in
// a normal CI unit test. This is therefore a SKELETON; fill in the TODOs to run it on a
// large-memory host, or convert it into a guard-assertion test once prepareParams adds
// the CPU_NODE_ASSERT recommended in the fix.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node test helpers used by other ov_cpu_unit_tests
//       (e.g. the Graph/Node single-layer test fixture under
//        src/plugins/intel_cpu/tests/unit/). Read that tree for exact symbol names.

TEST(GatherCpuNode, OffsetStrideDoesNotTruncateToInt32) {
    // TODO: build an ov::op::v8::Gather with a static data shape whose
    //       axisDim * afterAxisSize * dataTypeSize > INT32_MAX so that
    //       axisAndAfterAxisSizeInBytes overflows 32 bits.
    //   const ov::Shape dataShape = { /* beforeAxis */ 2, /* axisDim */ 0x20000001, 1 };
    //   const ov::Shape idxShape  = { 2 };
    //   const int64_t axis = 1;
    // TODO: compile the subgraph for the CPU plugin and invoke prepareParams/createPrimitive.
    //
    // Expected behaviour AFTER the fix (pick one, matching the chosen remediation):
    //   (a) guard variant: createPrimitive/prepareParams rejects the oversized stride.
    //       EXPECT_THROW(infer_request.infer(), ov::Exception);
    //   (b) 64-bit variant: offsets remain correct; no ASan heap-buffer-overflow and
    //       results match the reference Gather.
    //
    // Pre-fix, variant (b) run under ASan reports a heap-buffer-overflow READ inside
    // jitUniGatherKernel due to the negative wrapped dataBeforeAxisSumInBytes offset.
    GTEST_SKIP() << "Skeleton: requires >2GB data tensor or the prepareParams size guard; see TODOs.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter=GatherCpuNode.OffsetStrideDoesNotTruncateToInt32 . Pre-fix expectation under ASan: 'heap-buffer-overflow READ' originating in jitUniGatherKernel via the truncated p.dataBeforeAxisSumInBytes offset (gather.cpp:500/358-359). Post-fix: clean run (offsets 64-bit) or ov::Exception from the prepareParams stride guard.

## Suggested fix
Change `dataBeforeAxisSumInBytes`, `specIdxInBytes`, and `idxBatchSumInBytes` in `threadExecParams` (gather.h:39–43) to `std::vector<int64_t>`. Update the JIT kernel argument structs to match 64-bit pointer arithmetic. As a short-term mitigation, add a validation gate in `prepareParams`: `CPU_NODE_ASSERT(axisAndAfterAxisSizeInBytes <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), "tensor stride overflows int32");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #494.
