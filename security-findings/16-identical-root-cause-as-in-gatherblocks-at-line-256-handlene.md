# Security finding #16: Identical root cause as in `gatherBlocks`: at line 256 `HandleNegat…

**Summary:** Identical root cause as in `gatherBlocks`: at line 256 `HandleNegat…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Same as gatherBlocks path: OOB read from the source data buffer enabling heap/memory disclosure or crash. This path is taken for single-element (scalar slice) GatherND operations, which are common in practice.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather_nd.cpp:255` — `GatherND::GatherNDExecutor::gatherElementwise()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** User-supplied indices tensor at GATHERND_INDEXES port flowing through the OpenVINO inference boundary

## Description / Root cause
Identical root cause as in `gatherBlocks`: at line 256 `HandleNegativeIndices` returns an unchecked `int32_t index`, which at line 257 is used as `dataIdx += srcShifts[i] * index` with no upper-bound guard, then at line 259 `shiftedSrcData[dataIdx]` performs a direct element read using the unchecked offset. The `int32_t` is implicitly converted to `size_t` when multiplied by `srcShifts[i]` (size_t), meaning even a moderate positive OOB value (e.g., 200 with dimension 5) produces a large unchecked offset.

**Validator analysis:** CWE-125 Out-of-bounds Read is accurate. HandleNegativeIndices (gather_nd.cpp:272-278) only normalizes negatives (index += srcDims[...]) and returns the value without any upper-bound check; it leaves negatives that overshoot the dim still negative and leaves all over-large positives unchecked. In gatherElementwise (taken when dataLength==1, dispatched via exec()'s OV_SWITCH at lines 169/174), line 257 computes dataIdx += srcShifts[i]*index and line 259 dereferences shiftedSrcData[dataIdx], an OOB element read of the source buffer. indices is the user-supplied GATHERND_INDEXES runtime tensor (getDataAs<int32_t>, line 234); prepareParams (110-124) only reads dims/strides and performs no value clamping, so a moderate positive index (>= dim) yields a far out-of-bounds offset — memory disclosure or crash. Same root cause as the gatherBlocks path. The proposed fix (bounds check inside HandleNegativeIndices, throwing if index<0 || index>=srcDims[idx+batchDims]) is correct and closes BOTH paths with one change; minor caveat: the throw fires inside a parallel_nt lambda, but OpenVINO propagates exceptions out of its parallel sections so this surfaces as an ov::Exception at the inference boundary, which is acceptable. A slightly cheaper alternative would be to validate indices once before the parallel loop, but the proposed fix is sufficient.

## Exploit / Proof of Concept
Supply a GatherND model with `dataLength == 1` (so `gatherElementwise` is called) and an indices tensor with any positive value >= the corresponding data dimension. `HandleNegativeIndices` returns the raw value; `dataIdx` accumulates an offset past the end of `srcData`; `shiftedSrcData[dataIdx]` reads the out-of-bounds element. The `shiftedSrcData` pointer is only offset by the valid batch stride (line 248), giving no protection.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in GatherND::GatherNDExecutor::gatherElementwise
// Unchecked index path: gather_nd.cpp:256-259 (dataIdx via HandleNegativeIndices, gather_nd.cpp:272-278).
// Pre-fix: with dataLength==1 (scalar slice) and an indices value >= the data dimension,
//          shiftedSrcData[dataIdx] reads past the source buffer -> ASan heap-buffer-overflow.
// Post-fix: HandleNegativeIndices throws ov::Exception for out-of-range index, so inference
//           must reject the bad input (ASSERT_ANY_THROW) instead of reading OOB.
//
// NOTE: GatherNDExecutor and HandleNegativeIndices are PRIVATE nested members of
// ov::intel_cpu::node::GatherND and cannot be instantiated directly from the test
// harness. The flaw must therefore be exercised end-to-end by compiling a GatherND
// subgraph on the CPU plugin and running inference with a crafted out-of-range index.
// This is a SKELETON: the exact subgraph-builder / infer helpers must be filled from the
// surrounding ov_cpu_unit_tests tree (e.g. nodes/*_node_test.cpp) before use.

#include <gtest/gtest.h>
#include <vector>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(GatherND_CPU_OOB, ElementwiseIndexOutOfBoundsIsRejected) {
    // data shape [5] -> dataLength == 1 so gatherElementwise() path is taken.
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    // indices select a single scalar element; value 200 is far out of range for dim 5.
    // TODO: confirm indices precision/shape accepted by GatherND v8 (sliceRank == last dim == 1).
    auto indices = op::v0::Constant::create(element::i32, Shape{1, 1}, std::vector<int32_t>{200});
    auto gnd = std::make_shared<op::v8::GatherND>(data, indices, /*batch_dims=*/0);
    auto model = std::make_shared<Model>(OutputVector{gnd}, ParameterVector{data}, "gathernd_oob");

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{5});
    auto* p = in.data<int32_t>();
    for (int i = 0; i < 5; ++i) p[i] = i;
    req.set_input_tensor(in);

    // Pre-fix: ASan flags heap-buffer-overflow inside gatherElementwise (gather_nd.cpp:259).
    // Post-fix: HandleNegativeIndices throws ov::Exception, surfaced here as a thrown exception.
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests --config Release (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=GatherND_CPU_OOB.ElementwiseIndexOutOfBoundsIsRejected . Expected pre-fix: AddressSanitizer: heap-buffer-overflow READ in ov::intel_cpu::node::GatherND::GatherNDExecutor::gatherElementwise (gather_nd.cpp:259). Expected post-fix: the infer() call throws ov::Exception ("GatherND: index 200 is out of bounds...") and the test passes. TODO: verify the GatherND subgraph builder + infer helpers against an existing nodes/*_node_test.cpp before relying on this.

## Suggested fix
Apply the same fix as for `gatherBlocks`: in `HandleNegativeIndices` (line 272), after the negative normalization block and before the return at line 277, add: `if (index < 0 || static_cast<size_t>(index) >= srcDims[idx + batchDims]) OPENVINO_THROW("GatherND: index ", index, " is out of bounds for dimension ", srcDims[idx + batchDims]);`. This single fix closes both code paths.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #16.
