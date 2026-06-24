# Security finding #276: The 1D fast-path block (lines 868–886) casts `indicesPtr` directly …

**Summary:** The 1D fast-path block (lines 868–886) casts `indicesPtr` directly …

**CWE IDs:** CWE-129: Improper Validation of Array Index / CWE-787: Out-of-bounds Write
**Severity / Impact:** Out-of-bounds write to heap memory controlled by attacker-supplied index values. On a 32-bit or 64-bit heap layout, writing arbitrary `int32_t` values at arbitrary offsets relative to a heap buffer enables heap metadata corruption and is a strong primitive for achieving arbitrary code execution. Any application that loads and executes an untrusted ONNX model containing a 1D ScatterUpdate node (shape ≤ 64, i32 data/indices) is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883` — `ScatterUpdate::execute()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** INDICES input tensor element values supplied externally (e.g., from a malicious ONNX model graph); values flow into `indicesPtr` and are reinterpreted as `int32_t*` at line 880 without any sanitisation.

## Description / Root cause
The 1D fast-path block (lines 868–886) casts `indicesPtr` directly to `int32_t* pindices` and uses each element as an array index into `pdst` at line 883 (`pdst[pindices[i]] = pupdate[i]`) with no bounds check. `pindices[i]` is a signed `int32_t`; a negative value writes before the start of `pdst`, and any value ≥ `srcLength` (max 64) writes past the end. The general code path at lines 907–917 enforces `idxValue >= 0 && idxValue < srcDimAxis` via `CPU_NODE_ASSERT`, but the fast-path intentionally bypasses this validation entirely.

**Validator analysis:** Confirmed real and reachable. The 1D fast-path (scatter_update.cpp:868-886) is entered when mode==ScatterUpdate, src is 1D i32 with len<=64, indices 1D i32. Inside, lines 882-884 loop over updateCnt and write pdst[pindices[i]] where pindices[i] is a raw, signed, attacker-supplied int32 with NO validation, unlike the general path (911-916) that asserts 0<=idx<srcDimAxis. A negative index writes before pdst; an index >=srcLength writes past the i32 output buffer — a clear CWE-129/CWE-787 OOB write. Index values are tensor data, so shape inference cannot bound them; nothing upstream sanitizes them before this path. vulnType (CWE-129/CWE-787) and impact (controlled heap OOB write, exploitability up to potential RCE) are accurate, though absolute arbitrary write is constrained because the offset is i32*4 bytes from a freshly allocated output buffer — still a strong corruption primitive. The proposed fix is correct and sufficient: add CPU_NODE_ASSERT(idx>=0 && static_cast<size_t>(idx)<srcLength) inside the loop before the write (note: use srcLength, not srcDimAxis, since this path indexes the flat 1D dst). Slightly better: hoist the check out only if you precompute, but per-iteration assert mirrors the existing general-path semantics and is adequate.

## Exploit / Proof of Concept
Craft an ONNX model with a ScatterUpdate node where: DATA is a 1D int32 tensor of length ≤ 64, INDICES is a 1D int32 tensor whose element value is, e.g., -1 (negative) or 10000 (far past 64). When `execute()` is called: (1) fast-path condition at line 868 is satisfied; (2) `pindices[0]` reads the attacker-supplied -1 or 10000; (3) `pdst[-1]` or `pdst[10000]` is written with `pupdate[0]`, corrupting heap memory outside the allocated output buffer. No upstream check rejects out-of-range index values on this path.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for OOB write at
// targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883
// The 1D i32 fast-path writes pdst[pindices[i]] with no bounds check.
// Pre-fix: an out-of-range index (negative or >= srcLength) triggers an
// ASan heap-buffer-overflow on the dst tensor write.
// Post-fix: ScatterUpdate::execute must reject the index (CPU_NODE_ASSERT ->
// ov::Exception), so the infer call throws instead of corrupting memory.
//
// Target: ov_cpu_unit_tests (intel_cpu). The exact subgraph-test helper symbols
// must be confirmed against intel_cpu/tests/unit/ before use.

#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test harness headers used by existing
// node tests under src/plugins/intel_cpu/tests/unit/ (e.g. the ov::Model /
// ov::op::v3::ScatterUpdate builder + a compiled infer-request helper).

TEST(ScatterUpdate1DFastPath, OutOfRangeIndexIsRejected) {
    // Build the exact fast-path shape: DATA 1D i32 len<=64, INDICES 1D i32,
    // UPDATE matching INDICES, AXIS = 0.
    const std::vector<int32_t> data(8, 0);          // srcLength = 8 (<=64)
    const std::vector<int32_t> indices = {10000};   // far past srcLength -> OOB
    const std::vector<int32_t> updates = {42};
    const std::vector<int64_t> axis = {0};

    // TODO: construct ov::op::v3::ScatterUpdate(data, indices, updates, axis),
    // wrap in ov::Model, core.compile_model(model, "CPU"), create infer request,
    // set the input tensors above.

    // The unchecked write at scatter_update.cpp:883 must now be guarded.
    // EXPECT_THROW(infer_request.infer(), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu subgraph/infer harness; assert "
                    "EXPECT_THROW(infer(), ov::Exception) for OOB index.";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=ScatterUpdate1DFastPath.OutOfRangeIndexIsRejected. With ASan enabled, the pre-fix binary reports 'heap-buffer-overflow WRITE' at scatter_update.cpp:883 (pdst[pindices[i]]); after the fix the infer call throws ov::Exception (CPU_NODE_ASSERT) and the test passes.

## Suggested fix
Add the same range check that the general path uses before the write at line 883. Inside the loop at line 882, validate: `int32_t idx = pindices[i]; CPU_NODE_ASSERT(idx >= 0 && static_cast<size_t>(idx) < srcLength, "indices value out of range in 1D fast-path"); pdst[idx] = pupdate[i];`. Alternatively, remove the fast-path's special-casing of validation and share the existing check at lines 907–917 by not returning early before validation when the fast-path is taken.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #276.
