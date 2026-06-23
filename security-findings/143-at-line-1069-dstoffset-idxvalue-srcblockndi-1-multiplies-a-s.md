# Security finding #143: At line 1069, `dstOffset += idxValue * srcBlockND[i + 1]` multiplie…

**Summary:** At line 1069, `dstOffset += idxValue * srcBlockND[i + 1]` multiplie…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound → CWE-787: Out-of-bounds Write
**Severity / Impact:** A crafted model's indices tensor can cause `cpu_memcpy(dstData + dstOffset, ...)` at line 1079 to write to an arbitrary location beyond the destination buffer — heap or mapped memory — enabling heap corruption that may lead to remote code execution or a reliable crash (DoS). Any caller loading an untrusted ONNX or OpenVINO IR model is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:1069` — `ScatterUpdate::scatterNDUpdate(const MemoryPtr&, const MemoryPtr&, const MemoryPtr&, scatter_reductions::ReduceNone const&)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Indices tensor values read from a crafted ONNX/OpenVINO model at inference time (attacker-controlled)

## Description / Root cause
At line 1069, `dstOffset += idxValue * srcBlockND[i + 1]` multiplies a signed `int64_t` (`idxValue`, already adjusted for negative at line 1067) by an unsigned `size_t` (`srcBlockND[i+1]`). Per C++ arithmetic conversion rules, `idxValue` is implicitly converted to `uint64_t`, so a large positive `idxValue` (e.g. 2^33) multiplied by a large stride can wrap modulo 2^64 to a small `uint64_t`. This wrapped small value is then added into `dstOffset` (also `size_t`). The only bounds guard is the post-loop `CPU_NODE_ASSERT(dstOffset < elementsCount, ...)` at line 1073, which a wrapped (artificially small) `dstOffset` trivially satisfies. There is no per-element `0 <= idxValue < srcDataDim[i]` range check anywhere in this function.

**Validator analysis:** The defect is real: scatter_update.cpp lines 1063-1069 perform `dstOffset += idxValue * srcBlockND[i+1]` where idxValue is converted to uint64_t before the multiply, and there is NO per-element range validation (negative values are merely wrapped by `idxValue += srcDataDim[i]` at 1067, with no upper-bound check). The sole post-loop guard at 1073 only checks `dstOffset < elementsCount`. CWE-190→CWE-787 is the correct categorisation in spirit. Two caveats on the report's exploit: (1) the headline single-dimension example (product wrapping to 4) lands at an in-bounds-but-wrong offset, which is silent data corruption, NOT an OOB write; (2) the genuine OOB requires k < data rank so srcBlockND[k] > 1: then a crafted (possibly wrapped) idxValue can drive dstOffset to ~elementsCount-1, pass the start-only assertion, and the `sizeToUpdate = srcBlockND[k]*dataSize` byte copy at line 1079 spills past the buffer end — a real heap OOB write. So impact (OOB write / heap corruption / DoS) is accurate, though the specific PoC understates the alignment constraints. The proposed fix (insert `CPU_NODE_ASSERT(idxValue >= 0 && static_cast<size_t>(idxValue) < srcDataDim[i], ...)` right after the negative-normalization at line 1067) is correct and sufficient: it eliminates both the unsigned-wrap and the end-spill because per-dimension validity guarantees dstOffset+srcBlockND[k] <= elementsCount. The same unchecked pattern recurs in the reduction overload starting at line 1107-1109, so the fix should be applied there too for completeness.

## Exploit / Proof of Concept
Construct a ScatterNDUpdate model whose data tensor has shape [N] (e.g. N=100) and whose indices tensor supplies a single k-tuple entry where one index coordinate is a large positive int64 (e.g. idxValue = 2^33 + 1) and srcBlockND[i+1] = 4, so the product 4*(2^33+1) overflows uint64_t to a small value (e.g. 4). With dstOffset=4 < elementsCount=100 the assertion passes, and `cpu_memcpy(dstData + 4, update, sizeToUpdate)` executes with attacker-controlled source data at an attacker-chosen low byte-offset — but the true logical write would have been far outside the allocation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB write in ScatterUpdate::scatterNDUpdate (ReduceNone)
// Cited unchecked code: scatter_update.cpp:1063-1069 (no per-element range
// check on idxValue) guarded only by scatter_update.cpp:1073
// (`dstOffset < elementsCount`, start-offset only). The fix adds
// `CPU_NODE_ASSERT(idxValue >= 0 && (size_t)idxValue < srcDataDim[i], ...)`.
//
// This assertion encodes: a ScatterNDUpdate whose indices tensor carries an
// out-of-range / overflow-inducing coordinate must be REJECTED (throw), not
// silently executed. Pre-fix this triggers an ASan heap-buffer-overflow at the
// cpu_memcpy on scatter_update.cpp:1079; post-fix it throws ov::Exception.
//
// SKELETON: exact intel_cpu unit-test fixture/helpers for constructing and
// inferring a single-node ScatterNDUpdate graph must be copied from the
// surrounding tests under intel_cpu/tests/unit/ — symbols below are placeholders.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_nd_update.hpp"

using namespace ov;

TEST(intel_cpu_ScatterNDUpdate, RejectsOutOfRangeIndex_OOBWrite) {
    // data shape [4, 8] -> elementsCount = 32, srcBlockND[k=1] = 8 (>1 so a
    // spill past the end is a true OOB write, not just a wrong in-bounds write).
    // TODO: build a single ScatterNDUpdate op (reduction == NONE) via the
    //       intel_cpu test graph helper used in intel_gpu/intel_cpu unit tests.
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8});
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto scatter = std::make_shared<op::v3::ScatterNDUpdate>(data, indices, updates);
    auto model   = std::make_shared<Model>(OutputVector{scatter},
                                           ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // TODO: fill data/updates with valid buffers.
    // Craft the malicious index: a large positive int64 whose product with the
    // stride either wraps uint64 or drives dstOffset to ~elementsCount-1 so the
    // 8-element copy spills past the 32-element destination.
    Tensor idxT(element::i64, Shape{1, 1});
    idxT.data<int64_t>()[0] = static_cast<int64_t>((1LL << 33) + 1); // overflow / OOB driver
    req.set_input_tensor(1, idxT);

    // Pre-fix: ASan heap-buffer-overflow inside cpu_memcpy (scatter_update.cpp:1079).
    // Post-fix: CPU_NODE_ASSERT rejects the index -> ov::Exception.
    EXPECT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or ov_cpu_func_tests for the infer-request path). Run: ./ov_cpu_unit_tests --gtest_filter='intel_cpu_ScatterNDUpdate.RejectsOutOfRangeIndex_OOBWrite'. Pre-fix expectation (ASan build): heap-buffer-overflow WRITE in cpu_memcpy via ScatterUpdate::scatterNDUpdate (scatter_update.cpp:1079); post-fix expectation: test passes because the added per-element CPU_NODE_ASSERT (after scatter_update.cpp:1067) throws ov::Exception, satisfying EXPECT_ANY_THROW. NOTE: a crafted single-node model fixture is required; replace the TODO graph-builder placeholders with the intel_cpu unit-test helpers before compiling.

## Suggested fix
Before the accumulation at line 1069, validate each `idxValue` against the corresponding data dimension: after negative normalization (line 1067), assert `idxValue >= 0 && static_cast<size_t>(idxValue) < srcDataDim[i]`. This replaces the weak post-accumulation unsigned comparison with a strong per-element range check. E.g., insert after line 1067: `CPU_NODE_ASSERT(idxValue >= 0 && static_cast<size_t>(idxValue) < srcDataDim[i], " indices value out of range for dimension ", i);` Then the accumulation is provably in-bounds and the overflow cannot occur.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #143.
