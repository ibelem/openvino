# Security finding #492: At lines 655–657, the expression `((start + i + idxElPerVec) / (spe…

**Summary:** At lines 655–657, the expression `((start + i + idxElPerVec) / (spe…

**CWE IDs:** CWE-197: Numeric Truncation Error → CWE-125: Out-of-bounds Read
**Severity / Impact:** The truncated (sign-flipped or otherwise wrong) values are passed verbatim as `arg.beforeAxisDiff = p.srcBeforeAxisDiff.data()` at execute():511, where the JIT kernel uses each entry as a signed byte offset for loads from the source data buffer. An attacker-crafted model can direct the JIT kernel to load from an arbitrary offset relative to the source buffer pointer, giving a controlled out-of-bounds read. This affects every caller of the OpenVINO CPU Gather node with attacker-supplied models.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:655` — `Gather::initShortParams()`
**Validated for repos:** openvino
**Trust boundary:** Model-supplied tensor dimensions → prepareParams → initShortParams → JIT kernel byte offsets

## Description / Root cause
At lines 655–657, the expression `((start + i + idxElPerVec) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes - ((start + i) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes` is computed entirely in `uint64_t` arithmetic (all operands are `uint64_t` class fields), but is then implicitly truncated to `int` when stored into `p.srcBeforeAxisDiff[i]` (declared `std::vector<int>` in gather.h:41). `axisAndAfterAxisSizeInBytes = axisDim * afterAxisSize * dataTypeSize` (prepareParams:424,427); for any model where this product exceeds 2^31, the truncation silently wraps the stored offset.

**Validator analysis:** The CWE-197→CWE-125 truncation is REAL: axisAndAfterAxisSizeInBytes = axisDim*afterAxisSize*dataTypeSize is correctly accumulated in uint64_t (gather.cpp:427), the diff at 656-657 is computed in uint64_t, then implicitly narrowed into int srcBeforeAxisDiff[i] (gather.h:41) and passed to the kernel as arg.beforeAxisDiff (gather.cpp:511) — a signed byte offset into the source buffer. No INT_MAX guard exists. However the finding's EXPLOIT NUMBERS are WRONG: afterAxisSize=134217728 can never reach this code because the blocked branch returns early at gather.cpp:637 when afterAxisSize > idxElPerVec (≤16 for fp32 AVX-512). The truncation is therefore only reachable with a SMALL afterAxisSize (2..16) combined with a very large axisDim (~33M+ for fp32), i.e. a ~2GB+ embedding/data tensor — large but a constructible/valid model (e.g. large-vocab embedding gather). So the defect is real and reachable, though the impact characterization should be tempered: it requires a multi-GB source tensor, and the truncated offset is only partially attacker-controlled (a multiple of axisAndAfterAxisSizeInBytes). The proposed fix is on the right track: the simplest sufficient mitigation is the prepareParams-time guard CPU_NODE_ASSERT(axisAndAfterAxisSizeInBytes <= INT32_MAX, ...) since the JIT kernel consumes srcBeforeAxisDiff as int*. Widening srcBeforeAxisDiff to int64_t alone is NOT sufficient without also widening the JIT kernel's beforeAxisDiff argument and its load logic; the guard is the safer, complete fix. Note arg.axisAndAfterAxisSizeB (gather.cpp:493) is already passed as a uint64_t*, confirming the kernel mixes 64-bit and 32-bit views of the same quantity.

## Exploit / Proof of Concept
Supply a model with a Gather node whose data tensor has large trailing dimensions so that `afterAxisSize * dataTypeSize * axisDim > 2^31` (e.g., `afterAxisSize=134217728` for float32 data makes `axisAndAfterAxisSizeInBytes = axisDim * 536870912`). When `axisDim >= 4`, `axisAndAfterAxisSizeInBytes >= 2^31`; the quotient at line 656 multiplied by this value overflows past INT_MAX when cast to `int`, producing a small negative or near-zero value. The JIT kernel then fetches data from before the source buffer, leaking prior heap contents or crashing.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression target for gather.cpp:655-657 (uint64_t -> int truncation of
// srcBeforeAxisDiff when axisAndAfterAxisSizeInBytes > INT32_MAX).
// Harness: ov_cpu_unit_tests (gtest), test lives under
//   openvino/src/plugins/intel_cpu/tests/unit/
//
// SKELETON: a faithful end-to-end trigger needs a Gather data tensor whose
// axisDim*afterAxisSize*dataTypeSize > 2^31 (~2GB allocation) with
// afterAxisSize in [2..dataElPerVec], which is infeasible to allocate in a
// unit test. Encode the fix instead as a guard check on the computed scalar.
//
// TODO: replace the placeholder include/helper names with the real ones from
//       intel_cpu/tests/unit/ once confirmed by reading that tree.
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

namespace {
// Mirrors the offending expression's pre-store value (gather.cpp:656-657).
// k is the small quotient-difference (0/1) multiplied by the byte stride.
static int64_t computeBeforeAxisDiff(uint64_t k, uint64_t axisAndAfterAxisSizeInBytes) {
    return static_cast<int64_t>(k * axisAndAfterAxisSizeInBytes);
}
}

// Pre-fix: storing into std::vector<int> truncates; this test documents that
// the guard added by the fix must reject such a configuration.
TEST(GatherShortParams, SrcBeforeAxisDiffMustNotTruncateToInt) {
    // afterAxisSize=16 (fp32 blocked short case), axisDim chosen so the byte
    // stride exceeds INT32_MAX.
    const uint64_t afterAxisSize = 16;
    const uint64_t dataTypeSize  = 4; // f32
    const uint64_t axisDim       = 34000000ULL; // > 2^31 / (16*4)
    const uint64_t axisAndAfterAxisSizeInBytes = axisDim * afterAxisSize * dataTypeSize;

    ASSERT_GT(axisAndAfterAxisSizeInBytes,
              static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));

    const int64_t full = computeBeforeAxisDiff(/*k=*/1, axisAndAfterAxisSizeInBytes);
    const int truncated = static_cast<int>(full); // what gather.h:41 does today

    // Demonstrates the silent corruption the fix must prevent.
    EXPECT_NE(static_cast<int64_t>(truncated), full)
        << "srcBeforeAxisDiff truncated from " << full << " to " << truncated;

    // TODO: once the prepareParams guard
    //   CPU_NODE_ASSERT(axisAndAfterAxisSizeInBytes <= INT32_MAX, ...)
    // is added, replace the above with an end-to-end build of a Gather node
    // (Node::prepareParams) and EXPECT throw, e.g.:
    //   EXPECT_THROW(buildAndPrepareGather(axisDim, afterAxisSize, dataTypeSize), ov::Exception);
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=GatherShortParams.SrcBeforeAxisDiffMustNotTruncateToInt . Pre-fix the EXPECT_NE passes (documents truncation); the real regression form (TODO) expects prepareParams to throw ov::Exception once the axisAndAfterAxisSizeInBytes<=INT32_MAX guard is added. End-to-end allocation-based trigger would require ~2GB and is not appropriate for the unit harness, hence the scalar-guard skeleton.

## Suggested fix
Change `srcBeforeAxisDiff` (and `specIdxDiff`, `beforeAxPermMask`, `afterAxPermMask`, `permIdxMask`, `dataBeforeAxisSumInBytes`) in `threadExecParams` from `std::vector<int>` to `std::vector<int64_t>`, and update all JIT kernel argument structures accordingly. Before writing, assert that the computed value fits in the target type: `OPENVINO_ASSERT(result >= INT32_MIN && result <= INT32_MAX, ...)`. Alternatively, add a `prepareParams`-time guard: `CPU_NODE_ASSERT(axisAndAfterAxisSizeInBytes <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), "axisAndAfterAxisSizeInBytes overflows int32");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #492.
