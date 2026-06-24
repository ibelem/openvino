# Security finding #271: At gather.h:92, `axisDim` is declared as `int` (32-bit signed). At …

**Summary:** At gather.h:92, `axisDim` is declared as `int` (32-bit signed). At …

**CWE IDs:** CWE-197: Numeric Truncation Error / CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** The corrupted `axisAndAfterAxisSizeInBytes` and `srcAfterBatchSizeInBytes` values are passed directly to the JIT kernel at execute() lines 493–494 as pointer-arithmetic parameters (`arg.axisAndAfterAxisSizeB` and `arg.srcAfterBatchSizeB`). The JIT kernel uses them to compute source-pointer offsets over the input tensor buffer. A wrapped (too-small) value causes the kernel to read far outside the allocated input buffer, leading to an out-of-bounds read / heap information disclosure, or a crash (DoS). Any caller (process, ONNX Runtime with OpenVINO EP) loading a malicious model with an oversized axis dimension is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:173` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model tensor shape (axis dimension) → CPU plugin Gather node fields → JIT kernel src-pointer arithmetic

## Description / Root cause
At gather.h:92, `axisDim` is declared as `int` (32-bit signed). At gather.cpp:164 and 418, it is assigned directly from `dataDims[axis]` where `dataDims` is of type `VectorDims` (std::vector<size_t>, 64-bit on x86-64). A dimension value > INT_MAX silently truncates to a negative int. Then at gather.cpp:173 and 427, `axisDim * afterAxisSizeInBytes` promotes the `int` to `uint64_t` (C++ usual arithmetic conversions). A negative `axisDim` becomes a near-maximal uint64_t value; multiplied by any non-zero `afterAxisSizeInBytes` (also uint64_t), the product silently wraps to a small value. Even with a positive but large `axisDim` and a large `afterAxisSizeInBytes`, the 64-bit unsigned product wraps similarly. No saturation or overflow check exists on either path (lines 163–176 and 416–437). The wrapped result is stored into `axisAndAfterAxisSizeInBytes` and immediately used in the next chain step at gather.cpp:175/429 (`srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes`), compounding the error.

**Validator analysis:** The narrowing is real: VectorDims is std::vector<size_t>, axisDim is a 32-bit int (gather.h:92), assigned directly at gather.cpp:164 and 418 with no magnitude check (the only assert at 411 validates axis index range, not the dimension value). A data-tensor dimension in [2^31, 2^32) truncates to a NEGATIVE int; at gather.cpp:173/427 the usual arithmetic conversions promote that negative int to a near-max uint64_t, and the multiply by afterAxisSizeInBytes wraps to a small/garbage value stored in axisAndAfterAxisSizeInBytes / srcAfterBatchSizeInBytes, which are passed by pointer to the JIT kernel at execute() (lines 493-494) as src-pointer stride parameters → OOB read/crash. So CWE-197/CWE-190 is accurate. Caveat on practicality: triggering the negative-axisDim window requires the Gather data tensor to actually be materialized at a dimension ≥2^31 (a ≥2GB allocation), and the researcher's exploit case (a) is wrong — shape [2^32+1] truncates to axisDim=1 (harmless), not the dangerous case; the genuinely dangerous values are dims 0x80000000..0xFFFFFFFF. The pure 64-bit-overflow-of-the-product path needs unallocatable sizes and is not realistic. The proposed fix is correct and sufficient: widen axisDim to int64_t/size_t (eliminates the sign-flip truncation) and add __builtin_mul_overflow / checked-int guards before the multiplications at 172-175 and 426-429 with a CPU_NODE_THROW on overflow.

## Exploit / Proof of Concept
Craft an ONNX model whose Gather data tensor has an axis dimension value that when stored in `int axisDim` either (a) truncates and becomes negative — e.g., shape = [2^32 + 1, ...] so `dataDims[axis]` = 0x100000001, truncated to `axisDim = 1` but axisDim = (int)0x100000001 = 1, or (b) is set to INT_MAX (2147483647) with afterAxisSizeInBytes ≥ 3 (three float32 elements) so `(uint64_t)(2147483647) * 12 = ~25.7 GB` stays valid, but with a moderately sized data type and afterAxisSize makes `betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes` overflow uint64_t to a small value. At execute() the JIT kernel then computes src-pointer offsets with the wrapped tiny value, striding into memory outside the tensor allocation. No bounds check is performed before passing these values to the JIT kernel.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-197/CWE-190 at gather.h:92 + gather.cpp:164,173.
// Pre-fix: a Gather data dim in [2^31, 2^32) truncates to a negative `int axisDim`,
//          and `axisDim * afterAxisSizeInBytes` (int->uint64_t) wraps to a tiny
//          stride that is handed to the JIT kernel at gather.cpp:493-494 -> OOB.
// Post-fix: axisDim is int64_t/size_t and the multiply is overflow-checked, so the
//          node construction/compilation rejects (throws) the oversized dimension.
//
// HARNESS: ov_cpu_unit_tests (gtest). TODO: confirm exact target name and the
// graph-builder helpers from src/plugins/intel_cpu/tests/unit/.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// TODO: This test allocates an enormous data tensor to reach the >INT_MAX axis
// dimension; on memory-constrained CI it must be gated/skipped. Prefer testing
// the arithmetic guard at a unit level if the helper is exposed.
TEST(GatherNodeOverflow, AxisDimExceedingIntMaxIsRejected) {
    // axis dimension 0x80000000 (2^31) -> truncates to negative int pre-fix.
    const size_t kHugeAxisDim = static_cast<size_t>(1) << 31;
    auto data = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{ static_cast<int64_t>(kHugeAxisDim), 3 });
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{ 1 });
    auto axis = op::v0::Constant::create(element::i32, Shape{}, { 0 });
    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{ gather }, ParameterVector{ data, indices });

    ov::Core core;
    // Pre-fix: compile/exec builds corrupted stride params -> ASan OOB / UB.
    // Post-fix: the dimension-magnitude / multiply-overflow guard throws.
    EXPECT_ANY_THROW({ auto compiled = core.compile_model(model, "CPU"); });
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ./ov_cpu_unit_tests --gtest_filter=GatherNodeOverflow.AxisDimExceedingIntMaxIsRejected. Pre-fix expectation under ASan: heap-buffer-overflow READ inside the Gather JIT kernel (src pointer offset uses wrapped axisAndAfterAxisSizeInBytes from gather.cpp:173), or no throw (test fails). Post-fix: compile_model throws ov::Exception from the Gather node's overflow guard and the test passes. TODO: confirm target name (ov_cpu_unit_tests) and adjust if intel_cpu uses a different gtest binary; gate the 2GB allocation behind a memory check.

## Suggested fix
1. Change `int axisDim` (gather.h:92) to `int64_t axisDim` (or `size_t`) to eliminate the truncation on assignment from `VectorDims` elements. 2. Before each multiplication at gather.cpp:173/427, add an explicit overflow guard, for example: `if (axisDim < 0 || (afterAxisSizeInBytes != 0 && static_cast<uint64_t>(axisDim) > std::numeric_limits<uint64_t>::max() / afterAxisSizeInBytes)) CPU_NODE_THROW("axis dimension overflow");`. Apply the same pattern before the chain step at lines 174–175 and 428–429. Similarly guard `axisAndAfterAxisSize = axisDim * afterAxisSize` at lines 172/426. Using a checked-integer library (e.g., SafeInt or __builtin_mul_overflow) for all these steps is preferable.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #271.
