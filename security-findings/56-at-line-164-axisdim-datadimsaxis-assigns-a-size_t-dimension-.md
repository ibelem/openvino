# Security finding #56: At line 164, `axisDim = dataDims[axis]` assigns a `size_t` dimensio…

**Summary:** At line 164, `axisDim = dataDims[axis]` assigns a `size_t` dimensio…

**CWE IDs:** CWE-197: Numeric Truncation Error
**Severity / Impact:** The JIT gather kernel receives corrupted per-element byte-offset strides (`axisAndAfterAxisSizeB`, `srcAfterBatchSizeB`). Using those strides to compute source pointers inside the SIMD kernel results in out-of-bounds reads from arbitrary heap memory regions, causing information disclosure or process crash. The `axisDim` pointer is also passed directly to the kernel for index bounds-checking; a negative value can suppress OOB index detection entirely, escalating to OOB writes.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX/IR model → shape inference → Gather node construction → initSupportedPrimitiveDescriptors

## Description / Root cause
At line 164, `axisDim = dataDims[axis]` assigns a `size_t` dimension value into `axisDim` which is declared as `int` (gather.h:92). For any axis dimension > INT_MAX (2147483647), this silently truncates (and potentially produces a negative value). The truncated `axisDim` is then used in the multiplication at line 172: `axisAndAfterAxisSize = axisDim * afterAxisSize` — here `int * uint64_t` implicitly sign-extends `axisDim` first, so a negative `axisDim` (from truncation) produces an astronomically large `uint64_t` stride value. This wrong value feeds `axisAndAfterAxisSizeInBytes` (line 173) and `srcAfterBatchSizeInBytes` (line 175), which are passed verbatim to the JIT kernel (execute:493-494) as memory stride arguments. The identical truncation and cascade occur in prepareParams at line 418 / 426-429.

**Validator analysis:** The cited code is accurate: `int axisDim` (gather.h:92) receives a `size_t` shape dimension at gather.cpp:164 and 418 with no range guard — the only CPU_NODE_ASSERTs nearby bound `axis`/`batchDims` (lines 138, 411), never `axisDim`'s magnitude. CWE-197 (Numeric Truncation) is the correct classification: a static data axis dim > INT_MAX truncates and, sign-extended in `int * uint64_t` at 172-173/426-427, yields a corrupted huge `axisAndAfterAxisSizeInBytes`/`srcAfterBatchSizeInBytes` stride and a negative `axisDim` (the latter is fed by pointer to the kernel for index bound checks). The downstream impact (OOB read in the SIMD kernel; possible suppression of index bound checks) is plausible and the kernel struct genuinely consumes `const int* axisDim`. SKEPTICAL CAVEAT on impact severity: practical reachability of an actual OOB at execute() requires the data tensor to truly have a static axis dimension above ~2.15 billion AND be allocated (multi-GB), so the realistic outcome is more often a crash/corrupted-stride than a controllable arbitrary-heap-read primitive; the description slightly overstates a clean exploit. Nonetheless the truncation is a real defect from untrusted model shapes. The proposed fix is on the right track but should be sharpened: widening `axisDim` to `uint64_t`/`size_t` alone is INSUFFICIENT because the JIT kernel field `gatherJitExecArgs::axisDim` is `const int*` (gather_uni_kernel.hpp:55) and is dereferenced as a 32-bit value inside the SIMD kernel — so a coordinated width change across the kernel ABI is required, OR (simpler and sufficient for safety) add an explicit guard `CPU_NODE_ASSERT(dataDims[axis] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()), ...)` at both assignment sites (164 and 418) to reject oversized axis dims before any stride math. The non-zero check the finding suggests is secondary. The guard-and-reject approach is the minimally invasive correct fix.

## Exploit / Proof of Concept
Craft an ONNX model with a Gather node whose axis dimension is set to, say, 2^31 + 1 (≈ 2.15 billion). When `dataDims[axis] = 2147483649` is assigned to `int axisDim`, it truncates to `-2147483647`. The subsequent `axisDim * afterAxisSize` (with afterAxisSize = 1) promotes the negative int to uint64_t → `0xFFFFFFFF80000001`. The kernel then uses this as a stride, computing a source pointer far beyond the actual tensor allocation, leading to heap OOB read (and potential write if reverse-indexed).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-197 numeric truncation in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:164 (and :418)
//   `int axisDim = dataDims[axis];`  // gather.h:92 declares axisDim as int
// Pre-fix: a data axis dimension > INT_MAX truncates to a negative int,
//   then sign-extends into uint64_t byte strides (gather.cpp:172-175/426-429)
//   that are handed to the JIT kernel (execute:491-494). The fix must reject
//   such an axis dim (CPU_NODE_ASSERT) or carry it as size_t end-to-end.
//
// This test exercises Gather::initSupportedPrimitiveDescriptors via the CPU
// node construction path with a data shape whose axis dim exceeds INT32_MAX.
// It asserts construction/desc-init throws (post-fix) instead of silently
// computing corrupted strides.
//
// NOTE: SKELETON — the exact intel_cpu unit-test harness symbols for directly
// instantiating a Gather node (GraphContext, dummy ov::op::v8::Gather with a
// huge PartialShape) must be confirmed against the surrounding test tree under
// src/plugins/intel_cpu/tests/unit/ before this will compile. Allocating a real
// 2^31+1-element tensor is NOT required: the truncation happens at shape-init
// time (initSupportedPrimitiveDescriptors), before any execution/allocation.

#include <gtest/gtest.h>
#include <limits>
#include <memory>

#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: include the intel_cpu node + graph-context headers used by the existing
// unit tests, e.g.:
//   #include "nodes/gather.h"
//   #include "graph_context.h"
// and any helper that builds a Node from an ov::Node (see tests/unit/).

using namespace ov;

TEST(GatherCpuNodeTest, RejectsAxisDimExceedingInt32Range) {
    // Data shape with axis (=0) dimension just past INT32_MAX.
    const size_t hugeDim = static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1;
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{static_cast<int64_t>(hugeDim), 1});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);

    // TODO: replace with the repo's actual node-construction helper, e.g.:
    //   auto ctx  = std::make_shared<intel_cpu::GraphContext>(/* config, ... */);
    //   auto node = std::make_shared<intel_cpu::node::Gather>(gather, ctx);
    //   EXPECT_THROW(node->initSupportedPrimitiveDescriptors(), ov::Exception);
    //
    // Pre-fix: no throw; axisDim truncates to a negative int and feeds
    //   axisAndAfterAxisSizeInBytes/srcAfterBatchSizeInBytes as bogus strides.
    // Post-fix: a CPU_NODE_ASSERT on dataDims[axis] <= INT32_MAX throws here.
    GTEST_SKIP() << "Wire up intel_cpu Gather node construction helper from "
                    "src/plugins/intel_cpu/tests/unit/ before enabling.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests, with -DENABLE_SANITIZER=ON for the OOB-read path). Run: ov_cpu_unit_tests --gtest_filter='GatherCpuNodeTest.RejectsAxisDimExceedingInt32Range'. Pre-fix expectation: with the data tensor actually populated and execute() reached, AddressSanitizer reports 'heap-buffer-overflow READ' inside the gather JIT kernel from the corrupted axisAndAfterAxisSizeB/srcAfterBatchSizeB stride; at the shape-init layer the bug is a silent negative axisDim. Post-fix expectation: initSupportedPrimitiveDescriptors throws ov::Exception ('axisDim exceeds int32 range') and the test passes. The skeleton's TODOs must be resolved (real GraphContext/Gather node helper symbols from tests/unit/) before it compiles.

## Suggested fix
Change `axisDim` from `int` to `uint64_t` (or `size_t`) in gather.h:92. Add an explicit range check after assignment: `CPU_NODE_ASSERT(dataDims[axis] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()), "axisDim exceeds int32 range");` — or better, change every downstream consumer (the JIT kernel struct field `axisDim`) to use `uint64_t` consistently. Also validate that `axisDim` is non-zero before passing it to the JIT kernel.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #56.
