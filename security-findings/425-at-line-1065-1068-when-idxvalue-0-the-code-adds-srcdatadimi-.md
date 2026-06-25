# Security finding #425: At line 1065-1068, when `idxValue < 0` the code adds `srcDataDim[i]…

**Summary:** At line 1065-1068, when `idxValue < 0` the code adds `srcDataDim[i]…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound / CWE-20: Improper Input Validation
**Severity / Impact:** Heap out-of-bounds write (or controlled arbitrary write within the data tensor) triggered by a crafted model at inference time. An attacker who can supply a model (or dynamically-shaped indices input) can overwrite heap metadata or adjacent allocations, potentially leading to remote code execution or reliable DoS. All callers of the ScatterNDUpdate CPU kernel are affected when reduction mode is NONE.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1063` — `ScatterUpdate::scatterNDUpdate (ReduceNone overload)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled indices tensor value read from ONNX/OV model at runtime — getIndicesValue() returns int64_t directly from model buffer at line 1064 with no prior per-component range validation

## Description / Root cause
At line 1065-1068, when `idxValue < 0` the code adds `srcDataDim[i]` to correct for negative indexing, but there is no subsequent check that `0 <= idxValue < (int64_t)srcDataDim[i]`. A very negative value (e.g. -2^62) with a small `srcDataDim[i]` (e.g. 5) remains negative after correction. Positive out-of-range values (e.g. 2^62) skip the correction branch entirely. Either way, at line 1069 — `dstOffset += idxValue * srcBlockND[i + 1]` — the signed `int64_t idxValue` is implicitly converted to `uint64_t` for the mixed-type multiplication (usual arithmetic conversions, §8.5.6/[conv.arith]). A large or negative-then-converted value multiplied by a non-trivial `srcBlockND[i+1]` wraps `size_t` arithmetic. Over a multi-iteration loop (k > 1), the attacker can pick per-dimension index values such that partial wraps cancel out and the final `dstOffset` is small (< `elementsCount`), bypassing the post-loop guard at line 1073. After the guard, line 1076 does `dstOffset *= dataSize`, which can itself overflow if `dstOffset` is near `SIZE_MAX / dataSize`, causing `dstData + dstOffset` on line 1079 to point entirely outside the allocation.

**Validator analysis:** The flaw is real. Lines 1063-1069: idxValue (int64_t) is added into dstOffset (size_t) with the only validation being negative-index correction (1067) and the post-loop guard dstOffset<elementsCount (1073). Two genuine problems: (1) idxValue is never bounded to [0,srcDataDim[i]) per ONNX/OV semantics, so a still-negative (after correction) or huge-positive value converts to uint64_t in idxValue*srcBlockND[i+1] and wraps mod 2^64; (2) more concretely, the guard only ensures dstOffset<elementsCount but the memcpy at 1079 writes sizeToUpdate = srcBlockND[k]*dataSize bytes. For valid aligned indices, dstOffset is always a multiple of srcBlockND[k] (every term is a multiple of srcBlockND[k]) so the max value is elementsCount-srcBlockND[k] and the copy fits. But when srcBlockND[k] is NOT a power of two, wraparound can yield an UNALIGNED dstOffset (e.g. elementsCount-1) that still passes the <elementsCount guard, after which the srcBlockND[k]*dataSize copy overruns the data buffer by up to (srcBlockND[k]-1)*dataSize bytes — a heap OOB write. The claim's specific 'dstOffset*=dataSize overflows near SIZE_MAX' sub-scenario is largely moot (the guard caps dstOffset<elementsCount, and elementsCount*dataSize is the real allocation so it cannot overflow), but the underlying CWE-20/CWE-190 OOB-write conclusion is correct. The proposed primary fix is correct and sufficient: insert, right after line 1067, `CPU_NODE_ASSERT(idxValue >= 0 && static_cast<size_t>(idxValue) < srcDataDim[i], ...)`. With per-component bounds enforced, idxValue*srcBlockND[i+1] <= srcDataDim[i]*srcBlockND[i+1] <= elementsCount (no wrap), dstOffset is aligned and <= elementsCount-srcBlockND[k], so the memcpy stays in bounds — the 128-bit/checked-mul suggestion becomes unnecessary. Note the identical pattern exists in the templated reduction overload at lines 1107-1118, which should get the same fix.

## Exploit / Proof of Concept
Craft an ONNX ScatterNDUpdate node where: data tensor shape = [D0, D1] (e.g. D0=4, D1=4, elementsCount=16), k=2, srcBlockND=[16,4,1]. For index tuple (v0, v1): choose v0 = large positive int64 such that (uint64_t)v0 * 4 wraps to W0 mod 2^64, then choose v1 = target - W0 where target in [0,16). The accumulated dstOffset = W0 + v1 = target < 16, passing CPU_NODE_ASSERT at line 1073. The actual write (cpu_memcpy at line 1079) lands at an attacker-chosen offset within dstData. To escalate to a write outside the buffer: choose dstOffset close to SIZE_MAX/dataSize so that dstOffset *= dataSize (line 1076) overflows to a near-zero byte offset, then add a second update tuple whose dstOffset is slightly above SIZE_MAX/dataSize to point past the end of the allocation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing per-component index bounds check in
// ScatterUpdate::scatterNDUpdate (ReduceNone) at
// targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1063-1079.
//
// Pre-fix: a ScatterNDUpdate with an out-of-range / wrap-inducing index value
// produces an unaligned dstOffset that passes the dstOffset<elementsCount guard
// (line 1073) yet drives a cpu_memcpy of srcBlockND[k]*dataSize bytes (line 1079)
// past the end of the data buffer -> heap OOB write (caught by ASan).
// Post-fix: the per-component assert (idxValue in [0,srcDataDim[i])) rejects the
// model, so inference throws ov::Exception instead of writing OOB.
//
// TODO: confirm exact target/helpers by reading
//   targets/openvino/src/plugins/intel_cpu/tests/unit/  (target: ov_cpu_unit_tests)
// and the ScatterNDUpdate single-layer test under
//   .../tests/functional/single_layer_tests/scatter_ND_update.cpp
// TODO: pick a data shape whose inner block srcBlockND[k] is NOT a power of two
//   (e.g. data shape {4,3}, indices last-dim k=1 -> srcBlockND[1]=3) and an
//   indices value crafted so that (uint64_t)idxValue*3 wraps to elementsCount-1.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node/graph test fixtures used by ov_cpu_unit_tests.

TEST(ScatterNDUpdate_ReduceNone, RejectsOutOfRangeIndexInsteadOfOobWrite) {
    // TODO: build an OV model: ScatterNDUpdate(data{4,3}, indices{1,1}, updates{1,3})
    //       with reduction == NONE.
    // TODO: set the single index component to a wrap-inducing int64 value V such
    //       that ((uint64_t)V * srcBlockND[1]) % 2^64 == elementsCount - 1 (= 11),
    //       i.e. an unaligned offset that still satisfies dstOffset < elementsCount.
    // TODO: compile for the CPU plugin and run inference inside the assertion.
    //
    // EXPECT_THROW(infer(model, inputs), ov::Exception);   // passes only after the
    //                                                       // per-component bounds fix
    GTEST_SKIP() << "Skeleton: fill in intel_cpu CPU-plugin fixture + crafted indices.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or scatter ND functional single-layer test under intel_cpu/tests). Run: ov_cpu_unit_tests --gtest_filter=ScatterNDUpdate_ReduceNone.RejectsOutOfRangeIndexInsteadOfOobWrite . Expected pre-fix: AddressSanitizer 'heap-buffer-overflow WRITE' inside cpu_memcpy (scatter_update.cpp:1079); expected post-fix: ov::Exception thrown from the per-component CPU_NODE_ASSERT and the test passes.

## Suggested fix
After the negative-index correction at line 1067, add a per-component bounds check before accumulating: `if (idxValue < 0 || static_cast<size_t>(idxValue) >= srcDataDim[i]) { CPU_NODE_THROW(...); }`. Additionally, perform the multiplication in a checked 128-bit or saturating type (or use `__builtin_mul_overflow` / a SafeInt helper) before adding to `dstOffset`, and verify the intermediate product does not exceed `elementsCount` itself before the addition. Remove the post-loop single-shot guard as the sole protection and replace with per-iteration validation.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #425.
