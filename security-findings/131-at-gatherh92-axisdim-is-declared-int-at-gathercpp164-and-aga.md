# Security finding #131: At gather.h:92, `axisDim` is declared `int`. At gather.cpp:164 (and…

**Summary:** At gather.h:92, `axisDim` is declared `int`. At gather.cpp:164 (and…

**CWE IDs:** CWE-197: Numeric Truncation Error / CWE-194: Unexpected Sign Extension → CWE-125: Out-of-bounds Read / CWE-787: Out-of-bounds Write
**Severity / Impact:** The overflowed `axisAndAfterAxisSizeInBytes` is passed as a stride pointer to the JIT kernel (execute:493, executeDynamicImpl:566) and used directly to compute `srcIdx` in execReference:950 (`srcIdx = c1 + axisAndAfterAxisSizeInBytes * i`). The resulting huge byte offset causes `cpu_memcpy` at line 954 to read from far past the source allocation (OOB read → crash or info leak) and write to far past the destination allocation (OOB write → memory corruption / potential RCE). Any user who loads an adversary-crafted ONNX/OpenVINO model containing a Gather node with an oversized axis dimension is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvino
**Trust boundary:** Adversary-controlled model file: data-tensor dimension at the gather axis flows through getInputShapeAtPort(GATHER_DATA).getDims()[axis] into int axisDim at model-load time

## Description / Root cause
At gather.h:92, `axisDim` is declared `int`. At gather.cpp:164 (and again at prepareParams:418) it is assigned directly from `dataDims[axis]` which has type `size_t` (uint64_t on x86-64). No range check exists. A model-supplied axis dimension > INT_MAX causes `axisDim` to truncate to a negative value. At lines 172-173 (and 426-427), `axisDim * afterAxisSize` and `axisDim * afterAxisSizeInBytes` have type `int * uint64_t`; C++ arithmetic conversions sign-extend the negative `int` to uint64_t, yielding a near-UINT64_MAX value stored in `axisAndAfterAxisSize` and `axisAndAfterAxisSizeInBytes`. Line 174-175 (428-429) then propagate these into `srcAfterBatchSize` and `srcAfterBatchSizeInBytes` (all uint64_t fields). No overflow assertion or saturation check is present anywhere in the chain.

**Validator analysis:** The defect is real and present in the cited code. `axisDim` is an `int` (gather.h:92) assigned directly from `dataDims[axis]` (a size_t from a model-controlled static shape) at gather.cpp:164 and gather.cpp:418, with no upper-bound check anywhere on the path (the only guards are batchDims/axis range checks at lines 127/138/411, none of which bound the dimension value). A data axis dimension > INT_MAX (e.g. 0x80000001) truncates to a negative int; at lines 173/427 `axisDim * afterAxisSizeInBytes` performs int→uint64 conversion that sign-extends the negative value to a near-UINT64_MAX product stored in `axisAndAfterAxisSizeInBytes`, which propagates into `srcAfterBatchSizeInBytes` (174-175/428-429). At execution, execReference (line 950) computes `srcIdx = c1 + axisAndAfterAxisSizeInBytes * i` and feeds it to cpu_memcpy (954) → OOB read of srcData / OOB write to dstData; the `idx < static_cast<size_t>(axisDim)` guard at 947 also becomes a comparison against a huge value when axisDim is negative, steering execution into the unsafe branch. So CWE-197/CWE-194 → CWE-125/CWE-787 is an accurate categorization. Caveat on practicality: the memory-unsafe operation occurs at inference (execReference), not at mere load, and triggering it needs a static data tensor whose axis dimension exceeds 2^31 with an actually-bound input buffer of that size (multi-GB) — large but attacker-feasible since both model and inputs are adversary-controlled; the truncation/overflow of the cached strides itself happens at compile/prepareParams time regardless. The proposed fix is correct and sufficient: widening `axisDim` to int64_t plus a `dataDims[axis] > int64 max` throw, AND the per-product overflow guards (the latter are essential because even a valid int64 dimension can overflow the uint64 product). I would additionally recommend bounding total element count against the actual allocated buffer size to also catch the dynamic-shape path at prepareParams (line 418).

## Exploit / Proof of Concept
Craft a model with a Gather node whose data tensor has axis dimension set to 0x80000001 (2^31+1). On load, `axisDim = (int)0x80000001 = -2147483647`. At line 173: `axisDim * afterAxisSizeInBytes` → uint64 sign-extension of -2147483647 = 0xFFFFFFFF80000001, times `afterAxisSizeInBytes` (e.g. 4) = 0xFFFFFFFE00000004 (~18 EB). Stored in `axisAndAfterAxisSizeInBytes`. execReference line 950: `srcIdx = c1 + 0xFFFFFFFE00000004 * 1` — an enormous offset. `cpu_memcpy(&srcData[srcIdx], …)` faults or reads attacker-reachable memory. The OOB write to `dstData[dstIdx]` at line 954 enables memory corruption.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-197/CWE-194 -> CWE-125/CWE-787 in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164 (and :418)
//   openvino/src/plugins/intel_cpu/src/nodes/gather.h:92  (`int axisDim`)
//
// Pre-fix: a Gather whose DATA input declares an axis dimension > INT_MAX is
// silently truncated to a negative `int axisDim`; lines 172-173 sign-extend it
// into a ~UINT64_MAX stride, leading to OOB access in execReference (line 950/954).
// Post-fix (widen to int64_t + bounds/overflow throw): compile_model must reject
// the oversized dimension with an ov::Exception BEFORE any inference/allocation.
//
// NOTE: This builds the graph with the DATA tensor as a *Parameter* (not a
// Constant) so no multi-GB buffer is allocated at compile time; the overflow
// is detected purely from the declared static shape during
// Gather::initSupportedPrimitiveDescriptors().
//
// TODO: confirm the exact intel_cpu unit-test target/harness and includes by
// reading src/plugins/intel_cpu/tests/unit/ ; symbol names below are best-effort.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

TEST(GatherCpuAxisDimTruncation, RejectsOversizedAxisDimension) {
    // axis dimension > INT_MAX -> would truncate to a negative `int axisDim`
    constexpr uint64_t kOversized = 0x80000001ULL;  // 2^31 + 1

    auto data = std::make_shared<ov::op::v0::Parameter>(
        ov::element::u8,
        ov::Shape{static_cast<size_t>(kOversized), 1});
    auto indices = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::Shape{1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto gather = std::make_shared<ov::op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{gather->output(0)},
        ov::ParameterVector{data, indices});

    ov::Core core;
    // Pre-fix: compiles fine (truncation is silent); first inference would OOB.
    // Post-fix: the bounds/overflow check throws here.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
```
**Build / run:** Build target: ov_cpu_unit_tests (intel_cpu plugin unit tests; confirm exact name under openvino/src/plugins/intel_cpu/tests/unit/). Run: ov_cpu_unit_tests --gtest_filter='GatherCpuAxisDimTruncation.RejectsOversizedAxisDimension'. Pre-fix expectation: with ASan, building/running inference on this model triggers a heap-buffer-overflow / SEGV inside Gather::execReference (cpu_memcpy at gather.cpp:954) from the sign-extended `axisAndAfterAxisSizeInBytes`; the compile-time EXPECT_THROW does NOT fire (silent truncation). Post-fix expectation: core.compile_model throws ov::Exception ("Gather: axis dimension too large") during Gather::initSupportedPrimitiveDescriptors and the test passes. TODO: if compile_model does not eagerly run initSupportedPrimitiveDescriptors in this harness, drive Node creation directly via the intel_cpu node-test fixtures and wrap with ASSERT_ANY_THROW.

## Suggested fix
Change `axisDim` from `int` to `int64_t` in gather.h (and all uses that receive it from `dataDims`). At the assignment site (lines 164 and 418), add a bounds check: `if (dataDims[axis] > static_cast<size_t>(std::numeric_limits<int64_t>::max())) OPENVINO_THROW("Gather: axis dimension too large");`. Additionally add overflow guards before each product: `if (axisAndAfterAxisSize > SIZE_MAX / afterAxisSize) OPENVINO_THROW(...)` (or use a saturating-multiply helper). This ensures the multiply chain never wraps and that strides stored as uint64_t fields accurately reflect the actual tensor layout.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #131.
