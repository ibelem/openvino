# Security finding #289: Every per-element index bounds check in `scatterElementsUpdate` (al…

**Summary:** Every per-element index bounds check in `scatterElementsUpdate` (al…

**CWE IDs:** CWE-129: Improper Validation of Array Index
**Severity / Impact:** Out-of-bounds read and write on the data tensor heap allocation in release builds. An attacker who controls the ONNX indices tensor can write arbitrary DataType values to arbitrary heap addresses (one write per out-of-range index entry), enabling heap metadata corruption and potentially remote code execution inside any process loading the crafted model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:605` — `ScatterUpdate::scatterElementsUpdate()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Runtime indices tensor values from an untrusted ONNX ScatterElementsUpdate model, forwarded by the OpenVINO EP

## Description / Root cause
Every per-element index bounds check in `scatterElementsUpdate` (all template overloads, including the ReduceMean specialization at lines ~734, 758, 793, 814) is expressed solely as `assert(idxValue < data_dim_size && idxValue >= 0)`. `assert()` is unconditionally compiled out when `NDEBUG` is defined (all release/optimized builds). The only non-assert check is the negative-wrap `if (idxValue < 0) { idxValue += data_dim_size; }` at lines 602–603, 624–625, 647–648, 666–667, etc., which silently passes values that are more negative than `-data_dim_size` or are non-negative but ≥ `data_dim_size`. The `CPU_NODE_ASSERT` at line 573 guards only the `axis` argument, not the per-element index values.

**Validator analysis:** The CWE-129 category is accurate and a real, reachable heap OOB write exists in release builds. However the researcher's PRIMARY exploit (positive index N >= data_shape[axis]) is ALREADY mitigated: execute() runs with axisRelaxed=true for ScatterElementsUpdate (ctor :123) and the loop at scatter_update.cpp:911-916 validates EVERY index element with CPU_NODE_ASSERT(idxValue < srcDimAxis && (idxValue >= 0 || mode==ScatterElementsUpdate)) — this fires in all build configs and rejects positive overflow before the typed kernel runs. The residual, genuinely-unguarded path is the NEGATIVE one: for ScatterElementsUpdate the '|| mode==ScatterElementsUpdate' clause accepts arbitrarily negative indices at :913, and the only normalization (idxValue += data_dim_size, :602-603/624-625/647-648/666-667 and the ReduceMean specialization) correctly handles only [-data_dim_size,-1]; a crafted value such as -(data_dim_size+1000) stays negative after normalization, the assert at :605/627/650/669/734/758/793/814 is NDEBUG-compiled-out, and dataPtr[offsets[0] + idxValue*dataBlock_axisplus1] indexes before the buffer -> OOB read/write. So the impact ('arbitrary write to arbitrary heap addresses' via large positive N) is OVERSTATED, but a real attacker-controlled negative-offset heap OOB write remains -> memory corruption / potential RCE. The proposed fix (replace each assert with CPU_NODE_ASSERT/OPENVINO_ASSERT(idxValue>=0 && idxValue<data_dim_size) AFTER normalization) is correct and sufficient: it closes the negative-underflow gap (and is harmlessly redundant with :913 for the positive bound) and fires in release builds. A tighter alternative is to extend the existing :913 validation loop to also reject negatives below -srcDimAxis for ScatterElementsUpdate, which avoids per-element checks in the hot kernel.

## Exploit / Proof of Concept
Craft an ONNX v12 ScatterElementsUpdate node whose indices tensor contains at least one value N where N >= data_shape[axis] (e.g., N = data_shape[axis] + 1000). In a release build (NDEBUG defined), the `assert` at line 627 is compiled away. The surviving code at line 628 computes `dst = &dataPtr[offsets[0] + N * dataBlock_axisplus1]`—`N * dataBlock_axisplus1` is an unsigned product that lands well past the buffer end—and the kernel at line 630 writes an update value to that address. Negative overflow is equally exploitable: supply idxValue = -(data_dim_size + 1); after `idxValue += data_dim_size` it is -1, a negative int64_t; multiplied by the size_t `dataBlock_axisplus1` it wraps to a huge unsigned offset, pointing into unrelated heap memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:605/627/650/669 (and ReduceMean :734/758/793/814).
// Pre-fix: a ScatterElementsUpdate whose indices contain a value more negative than
// -data_shape[axis] (e.g. -(dim+1000)) passes the upper-bound check at :913
// (negatives are allowed for ScatterElementsUpdate) and, because the per-element
// assert() is compiled out under NDEBUG, normalization (idxValue += data_dim_size)
// leaves a negative offset -> OOB heap write at :606/628 (ASan: heap-buffer-overflow
// / SEGV). Post-fix the CPU_NODE_ASSERT must reject the model with ov::Exception.
//
// Harness: ov_cpu_unit_tests (intel_cpu component, gtest). SKELETON — exact graph-build
// helpers / ScatterElementsUpdate op factory and the CPU graph-exec wrapper used by
// intel_cpu/tests/unit must be confirmed by reading the neighboring tests.
#include <gtest/gtest.h>
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/runtime/core.hpp"

TEST(scatter_elements_update_cpu, negative_index_below_neg_dim_is_rejected) {
    // TODO: build a tiny ov::Model with a v12 ScatterElementsUpdate node:
    //   data    : shape {4}, f32
    //   indices : shape {1}, i64, VALUE = -(4 + 1000)  // out of [-4,3]
    //   updates : shape {1}, f32
    //   axis    : 0
    // TODO: compile on "CPU" and infer; the crafted negative index must be rejected.
    // EXPECT_THROW(run_on_cpu(model, inputs), ov::Exception);   // passes only after fix
    GTEST_SKIP() << "TODO: wire up intel_cpu graph-exec helper + ScatterElementsUpdate factory";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=scatter_elements_update_cpu.negative_index_below_neg_dim_is_rejected . Pre-fix (NDEBUG, ASan build): heap-buffer-overflow WRITE in ScatterUpdate::scatterElementsUpdate at scatter_update.cpp:606/628 (negative computed offset). Post-fix: CPU_NODE_ASSERT throws ov::Exception and the EXPECT_THROW passes with no ASan report.

## Suggested fix
Replace every `assert(idxValue < data_dim_size && idxValue >= 0)` with a `CPU_NODE_ASSERT` (or `OPENVINO_ASSERT`) that fires in all build configurations, e.g.: `CPU_NODE_ASSERT(idxValue >= 0 && idxValue < data_dim_size, "Index value ", idxValue, " is out of range [0, ", data_dim_size, ").");` Insert this check immediately after the negative-normalization block at lines 602–604, 624–626, 647–649, 666–668, and all equivalent sites in the ReduceMean specialization (~733–735, 755–758, 790–793, 811–814). Alternatively, clamp or reject the entire indices tensor during model loading by validating the constant-fold-visible index range; for dynamic indices add a runtime loop over the indices buffer in `execute()` before dispatching to the typed kernel.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #289.
