# Security finding #495: At line 173, `axisDim` (declared as `int`, 32-bit signed, gather.h:…

**Summary:** At line 173, `axisDim` (declared as `int`, 32-bit signed, gather.h:…

**CWE IDs:** CWE-681: Incorrect Conversion Between Numeric Types / CWE-190: Integer Overflow
**Severity / Impact:** Corrupted `srcAfterBatchSizeInBytes` is passed directly to the JIT gather kernel as `arg.srcAfterBatchSizeB` (execute:494). The JIT kernel uses this value as a batch-stride in pointer arithmetic over the source data buffer. An overflowed value (e.g. ~UINT64_MAX or 0) causes the kernel to stride backward or to an attacker-controlled address, enabling an out-of-bounds read/write or crash (DoS/potential RCE). Affects any consumer loading an ONNX/IR model with large Gather axis dimensions.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:173` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-supplied tensor shape dimensions from untrusted model graph (GATHER_DATA input shape)

## Description / Root cause
At line 173, `axisDim` (declared as `int`, 32-bit signed, gather.h:92) is multiplied with `afterAxisSizeInBytes` (uint64_t). `axisDim` was assigned from `dataDims[axis]` at line 164 — a narrowing truncation from the underlying 64-bit Dim/size_t. If the model supplies axisDim > INT_MAX, the truncated value is negative; the implicit promotion of a negative `int` to `uint64_t` in the multiplication wraps to ~(UINT64_MAX - |val|+1), producing a near-UINT64_MAX stride. At line 175, a second unchecked uint64_t × uint64_t multiplication (`betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes`) can independently overflow and wrap, e.g. to 0. Both are stored in `srcAfterBatchSizeInBytes` with no overflow check.

**Validator analysis:** The core defect is real: axisDim is a signed 32-bit int (gather.h:92) assigned from a size_t shape dim (gather.cpp:164 and prepareParams:418), a genuine narrowing truncation (CWE-681). A data axis dim > INT_MAX truncates to a negative int; promotion to uint64_t in the multiply at 173 (and 427) yields a near-UINT64_MAX value with no overflow/range check, and srcAfterBatchSizeInBytes is used directly as a batch stride by the JIT kernel (execute:494, prepareParams paths). The line-175 pure uint64_t*uint64_t overflow is far less realistic (would require >2^64 total elements, i.e. an unallocatable tensor) — the credible vector is the int narrowing of axisDim. Realistic exploitability is tempered by the fact that running inference needs the multi-GB data buffer to actually exist, so the most likely outcome is OOM/DoS or a kernel OOB read rather than reliable RCE; the impact should be characterized as OOB read / DoS, RCE 'potential' only. The proposed fix is correct and largely sufficient: widen axisDim to int64_t/Dim (gather.h:92) and add explicit divide-based overflow guards before the chained multiplies at 172-175 and 426-429, ideally throwing via CPU_NODE_ASSERT. A stronger fix also bounds-checks axisDim against the data buffer element count at prepareParams before the kernel uses it. The fix must be applied to BOTH initSupportedPrimitiveDescriptors and prepareParams, which the finding correctly notes.

## Exploit / Proof of Concept
Craft a model with a Gather node whose data tensor has an axis dimension of, say, 2^31+1 (> INT_MAX). After `initSupportedPrimitiveDescriptors` line 164, `axisDim` = (int)(2^31+1) = -2147483647 (signed truncation). At line 173, `(-2147483647) * afterAxisSizeInBytes` — C++ promotes the negative int to uint64_t as 0xFFFFFFFF80000001, and the product overflows. `srcAfterBatchSizeInBytes` receives a corrupted stride. At execute:494 the JIT kernel loads this as `arg.srcAfterBatchSizeB` and uses it to advance a pointer, walking to an arbitrary address on the next batch iteration, causing OOB memory access.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-681/CWE-190 in intel_cpu Gather (gather.h:92, gather.cpp:164/173/427).
// Pre-fix: axisDim is a signed 32-bit int; a data-axis dim > INT_MAX truncates to a
// negative value and the unchecked axisDim*afterAxisSizeInBytes multiply wraps a uint64_t
// stride that the JIT kernel later uses for pointer arithmetic (OOB / corrupt stride).
// Post-fix: axisDim is widened (int64_t/Dim) and the chained shape-product multiplies are
// guarded, so building/compiling a Gather with such a shape must be rejected rather than
// silently producing a wrapped stride.
//
// SKELETON: a self-contained, compilable repro needs either a crafted IR/ONNX with a
// 2^31+1 Gather data-axis dimension or direct construction of the intel_cpu Gather node
// with a mocked >INT_MAX static data shape. Exact ov_cpu_unit_tests fixture/symbol names
// for driving a single node must be confirmed against the surrounding test tree.
#include <gtest/gtest.h>
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(GatherCpuShapeOverflow, AxisDimAboveIntMaxIsRejected) {
    // TODO: confirm the correct ov_cpu_unit_tests harness/helper for compiling a single
    // CPU node (see intel_cpu/tests/unit and the Subgraph/SingleLayer test utilities).
    constexpr int64_t kHugeAxis = (int64_t)std::numeric_limits<int>::max() + 2; // > INT_MAX

    auto data    = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{kHugeAxis, 1});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model   = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data, indices});

    Core core;
    // Pre-fix: compilation builds the node and computes a truncated/wrapped stride in
    // Gather::initSupportedPrimitiveDescriptors (gather.cpp:173) with no diagnostic.
    // Post-fix: the narrowing/overflow guard must reject the oversized axis dimension.
    // TODO: if compile_model does not surface the guard, drive Gather::prepareParams
    // directly and assert CPU_NODE_THROW fires.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=GatherCpuShapeOverflow.AxisDimAboveIntMaxIsRejected . Pre-fix expectation: no throw (test fails) and, under ASan with a real allocated buffer, a heap-buffer-overflow / wild pointer in the Gather JIT kernel due to the wrapped srcAfterBatchSizeInBytes stride; post-fix: ov::Exception thrown by the added overflow/narrowing guard.

## Suggested fix
1) Change `axisDim` from `int` to `int64_t` (or `uint64_t`) in gather.h line 92 and update all uses consistently. 2) Add explicit overflow checks before each chained multiplication, e.g.: `if (axisDim > 0 && afterAxisSizeInBytes > 0 && static_cast<uint64_t>(axisDim) > std::numeric_limits<uint64_t>::max() / afterAxisSizeInBytes) CPU_NODE_THROW("shape product overflow");` and similarly for the `betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes` product at line 175. Apply the same guards at prepareParams lines 427/429.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #495.
