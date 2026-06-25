# Security finding #422: The 1D fast-path at line 883 (`pdst[pindices[i]] = pupdate[i]`) use…

**Summary:** The 1D fast-path at line 883 (`pdst[pindices[i]] = pupdate[i]`) use…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Arbitrary out-of-bounds write to heap memory adjacent to the destination output buffer. A negative index (e.g., -1) writes before the buffer; a large positive index (e.g., INT32_MAX) writes far past it. On a typical heap layout this can corrupt allocator metadata, vtable pointers, or other live objects — exploitable for remote code execution if the model file is loaded from an untrusted source (e.g., ML model marketplace, user-supplied ONNX file).
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883` — `ScatterUpdate::execute()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Indices tensor values embedded in an ONNX/OV model file, deserialized and passed to the CPU inference engine execute() without sanitization

## Description / Root cause
The 1D fast-path at line 883 (`pdst[pindices[i]] = pupdate[i]`) uses `pindices[i]` — a raw `int32_t` read from an attacker-controlled tensor — directly as an array index into `pdst` with zero bounds checking. The destination buffer `pdst` has at most `srcDataDim[0]` (≤64) `int32_t` elements, but there is no check that `pindices[i]` is in `[0, srcDataDim[0]-1]`. The general-path validation at lines 911-916 (which asserts `idxValue < srcDimAxis && idxValue >= 0`) is skipped entirely because of the early `return` at line 885.

**Validator analysis:** Confirmed: the 1D fast-path (entered at line 868 when scatterUpdateMode==ScatterUpdate, srcDataDim 1D and <=64, indicesDim<=1, indicesPrec/dataPrec==i32) at line 883 uses pindices[i] — an int32 value read straight from the INDICES_ID tensor — as the index into pdst with no range check, then returns at 885 bypassing all validation. The general (axisRelaxed) path at 911-916 enforces idxValue in [0, srcDimAxis); the fast path enforces nothing. A negative or >=srcDataDim[0] index yields a heap OOB write of 4 bytes, so CWE-787 is the correct classification and the RCE/heap-corruption impact is plausible. Reachability depends on the indices tensor being attacker-controllable: if fed by a Constant in the model (or a controllable input) the value is fully under attacker control, satisfying the stated trust boundary. The proposed fix is correct and sufficient: add `int32_t idx = pindices[i]; if (idx < 0 || static_cast<size_t>(idx) >= srcLength) OPENVINO_THROW(...); pdst[idx] = pupdate[i];` inside the loop (use srcLength==srcDataDim[0], which is the actual pdst element count). Reusing CPU_NODE_ASSERT keeps it consistent with the general path. Note also updateCnt comes from updateDims[0] which should be validated against indices count, but the index-bound check is the essential fix.

## Exploit / Proof of Concept
Craft an ONNX ScatterUpdate node where the data tensor is shape [1] (satisfies `srcDataDim[0] <= 64`), dataPrec=int32, indicesPrec=int32, and the indices tensor contains the value -1 or any value ≥ 64. When the model is loaded and executed via OpenVINO CPU EP, `execute()` enters the fast-path at line 868, skips all validation, and executes `pdst[-1] = pupdate[0]`, writing 4 bytes immediately before the destination heap buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-787 OOB write at
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:882-884
// The 1D fast-path writes pdst[pindices[i]] with no bounds check. A ScatterUpdate
// with data shape [1] (i32), indices=[-1] (or >=srcDataDim[0]) hits the fast-path
// (line 868) and writes out of bounds, returning at line 885 before any validation.
// Pre-fix: ASan heap-buffer-overflow on the dst allocation.
// Post-fix: index is range-checked and execution throws ov::Exception.
//
// TODO: place under openvino/src/plugins/intel_cpu/tests/unit/ and build with
//       ov_cpu_unit_tests. Confirm exact node/test helpers by reading the existing
//       intel_cpu unit test tree (e.g. nodes/ helpers) before use.
#include <gtest/gtest.h>
// TODO: include the intel_cpu node test fixtures / graph builder helpers used by
//       the existing ScatterUpdate unit tests (symbols unverified).

TEST(ScatterUpdateCpu, FastPathRejectsOutOfRangeIndex) {
    // TODO: build a ScatterUpdate node with:
    //   data    : i32, shape [1]  (srcDataDim[0] <= 64 -> enters fast path)
    //   indices : i32, shape [1], value = -1   (also test value = 64)
    //   update  : i32, shape [1]
    // and run execute().
    //
    // Expected after fix: ov::Exception (index out of range). Before fix this is
    // an undetected OOB write caught only by ASan.
    // EXPECT_THROW(run_scatter_update(/*data shape*/ {1}, /*indices*/ {-1}, /*update*/ {7}), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu ScatterUpdate node test fixture";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests ; run: ./ov_cpu_unit_tests --gtest_filter=ScatterUpdateCpu.FastPathRejectsOutOfRangeIndex . With -DENABLE_SANITIZER=ON, pre-fix expect AddressSanitizer 'heap-buffer-overflow WRITE of size 4' at scatter_update.cpp:883; post-fix the case throws ov::Exception and the test passes.

## Suggested fix
Add a bounds check inside the fast-path loop before the write. For example: `int32_t idx = pindices[i]; if (idx < 0 || static_cast<size_t>(idx) >= srcDataDim[0]) OPENVINO_THROW("ScatterUpdate: index out of range in 1D fast-path"); pdst[idx] = pupdate[i];`. Alternatively, reuse the existing `getIndicesValue` + `CPU_NODE_ASSERT` pattern already used in the general path (lines 912-915) before entering the fast-path write loop.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #422.
