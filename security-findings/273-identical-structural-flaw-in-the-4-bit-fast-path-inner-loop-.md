# Security finding #273: Identical structural flaw in the 4-bit fast-path: inner loop `for (…

**Summary:** Identical structural flaw in the 4-bit fast-path: inner loop `for (…

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-787: Out-of-bounds Write
**Severity / Impact:** Same as the 8-bit path: OOB read on packed 4-bit source data and OOB write to destination during GatherCompressed inference on models with non-divisible `afterAxisSize`/`scale_group_size`. The nibble-addressing `g >> 1` means each out-of-bounds step reads a byte that may belong to an unrelated tensor or beyond the allocation.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:735` — `Gather::execCompressed4Bit()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-supplied tensor shapes for GATHER_DATA and GATHER_SCALE flowing into scale_group_size and afterAxisSize without divisibility validation

## Description / Root cause
Identical structural flaw in the 4-bit fast-path: inner loop `for (size_t g = p; g < p + scale_group_size; g++)` at line 735 has no guard `g < srcIdx + afterAxisSize`. `srcData[g >> 1]` at line 736 and `pdst[dst_idx]` at line 737 both go out of bounds when `afterAxisSize % scale_group_size != 0` by the same mechanism as the 8-bit case.

**Validator analysis:** The 4-bit fast-path (cond1||cond2 branch, lines 730-741) is structurally identical to the 8-bit path (815-826) that is the source finding: the outer loop steps p by scale_group_size while p < srcIdx+afterAxisSize, and the inner loop unconditionally iterates g in [p, p+scale_group_size). When afterAxisSize is not a multiple of scale_group_size the final outer iteration overruns: it reads srcData[(srcIdx+afterAxisSize+k)>>1] (OOB read on packed nibble bytes) and writes pdst[dst_idx] past the afterAxisSize-sized destination region (OOB write). scale_group_size is derived by integer division (line 206) of GATHER_DATA elements by GATHER_SCALE elements, with no enforcement that it divides afterAxisSize anywhere in initSupportedPrimitiveDescriptors (191-234) or the constructor. With non-scalar scale (cond3 false) and a per-group scale whose count makes scale_group_size non-divisible (e.g. afterAxisSize=7, scale_group_size=3 -> outer p=6 inner reads g=7,8), the bug fires. CWE-125/CWE-787 classification is accurate. The proposed fix is correct: add `&& g < srcIdx + afterAxisSize` to the inner-loop condition at line 735 (and mirror in the 8-bit path at line 820); enforcing afterAxisSize % scale_group_size == 0 at load time is an equally valid, arguably cleaner guard, but the model may legitimately have ragged tails so clamping the loop is the safer minimal fix.

## Exploit / Proof of Concept
Same model crafting as the 8-bit case but with a u4/i4 data precision. With K=7 and scale_group_size=4: the final inner-loop iteration at g=srcIdx+7 reads `srcData[(srcIdx+7)>>1]`, a byte shared with or beyond the row boundary, and writes one extra element to `pdst`.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression for OOB in Gather::execCompressed4Bit, gather.cpp:735-737.
// Encodes: a GatherCompressed with u4 data where afterAxisSize is NOT a
// multiple of scale_group_size must NOT read/write out of bounds (caught by
// ASan pre-fix; passes once the inner loop is clamped to srcIdx+afterAxisSize).
//
// SKELETON: exact GatherCompressed builder + tensor wiring must be filled in
// from the existing intel_cpu single-layer test helpers; symbol names below
// are placeholders pending a read of the surrounding test tree.
#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(GatherCompressed4Bit, NonDivisibleScaleGroupSizeNoOOB) {
    // TODO: build a GatherCompressed graph with:
    //   - GATHER_DATA: shape [vocab=4, afterAxisSize=7], precision u4
    //   - GATHER_SCALE: shape [4, 2]  -> scale_group_size = 7/2 = 3 (int trunc),
    //     i.e. afterAxisSize(7) % scale_group_size(3) != 0
    //   - GATHER_INDICES: {0}, axis const = 0  (drives the axis==0 fast path)
    //   - have_scalar_scale == false so cond3 is false and the cond1||cond2
    //     inner-loop branch at gather.cpp:730-741 is taken.
    // TODO: compile for the CPU plugin and infer; pre-fix this triggers an
    //       ASan heap-buffer-overflow at gather.cpp:736 (srcData[g>>1]) and
    //       gather.cpp:737 (pdst[dst_idx]).
    // TODO: assert inference completes and output buffer is untouched beyond
    //       afterAxisSize elements per row.
    GTEST_SKIP() << "Fill in GatherCompressed builder from intel_cpu test helpers";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests, with -DENABLE_SANITIZER=ON). Run: ov_cpu_unit_tests --gtest_filter=GatherCompressed4Bit.NonDivisibleScaleGroupSizeNoOOB . Expected pre-fix: AddressSanitizer heap-buffer-overflow READ in Gather::execCompressed4Bit at gather.cpp:736 (srcData[g>>1]) and WRITE at gather.cpp:737 (pdst[dst_idx]); post-fix the test passes with no ASan report.

## Suggested fix
Add `g < srcIdx + afterAxisSize` to the inner-loop guard at line 735, mirroring the fix for the 8-bit path, or enforce `afterAxisSize % scale_group_size == 0` at model load time in `initSupportedPrimitiveDescriptors`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #273.
