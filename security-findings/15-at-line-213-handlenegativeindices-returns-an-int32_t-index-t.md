# Security finding #15: At line 213 `HandleNegativeIndices` returns an `int32_t index` that…

**Summary:** At line 213 `HandleNegativeIndices` returns an `int32_t index` that…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read from the source data buffer. With a crafted index tensor supplied as model input, an attacker can read arbitrary memory beyond the data tensor allocation, enabling information disclosure (e.g., heap metadata, adjacent buffers) or a process crash. Any application using the OpenVINO CPU EP with dynamic/runtime index tensors is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather_nd.cpp:212` — `GatherND::GatherNDExecutor::gatherBlocks()`
**Validated for repos:** openvino
**Trust boundary:** User-supplied indices tensor at GATHERND_INDEXES port flowing through the OpenVINO inference boundary

## Description / Root cause
At line 213 `HandleNegativeIndices` returns an `int32_t index` that has only been normalized for negative values (line 274-276) but never upper-bound-checked against `srcDims[idx+batchDims]`. This unchecked index is immediately used at line 214 as `dataIdx += srcShifts[i] * index` (where `srcShifts[i]` is `size_t`; a positive OOB `int32_t` index is widened to `size_t` silently), and the resulting `dataIdx` is used at line 216 in `cpu_memcpy(shiftedDstData, &(shiftedSrcData[dataIdx]), dataLength)` with no per-element bounds guard.

**Validator analysis:** vulnType CWE-125 Out-of-bounds Read is accurate: a user-supplied indices tensor element >= srcDims[i+batchDims] (or a still-negative value after normalization) is multiplied by srcShifts[i] and used as a byte offset into shiftedSrcData for cpu_memcpy of dataLength bytes, reading past the source allocation. The impact (info disclosure / crash from heap OOB read) is accurate; it is a read, not a write, so it is disclosure/DoS rather than RCE. Reachability is confirmed: GatherND v5/v8 accepts a runtime indices tensor, no upstream clamp exists (HandleNegativeIndices is the only normalization, and it does not upper-bound-check), and the same flaw affects gatherElementwise (line 257/259) which shares HandleNegativeIndices. The proposed fix is correct and sufficient: adding the upper-bound check inside HandleNegativeIndices fixes BOTH gatherBlocks and gatherElementwise in one place, and the condition `if (index < 0 || static_cast<size_t>(index) >= srcDims[idx + batchDims]) OPENVINO_THROW(...)` also catches the case where a negative index remains negative after adding srcDims (under-flow). OPENVINO_THROW at execute() time is acceptable because the inference API converts it to an error. One refinement: prefer throwing inside HandleNegativeIndices (single choke point) over the call-site guard, and ensure the check uses srcDims[idx + batchDims] (not srcDims[i] without batchDims offset) to match the per-slice dimension.

## Exploit / Proof of Concept
Supply a GatherND model where the indices tensor contains a value >= the corresponding data dimension (e.g., data shape [4,4], indices value 1000). No upstream validation in `prepareParams` or the executor constructor rejects positive-out-of-range index values. At inference time, `HandleNegativeIndices` returns 1000 unmodified, `dataIdx` accumulates `srcShifts[0] * 1000` (= stride_0 * 1000 bytes past the start), and `cpu_memcpy` reads dataLength bytes far outside the allocated source buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in GatherND CPU executor.
// Encodes the fix at gather_nd.cpp:272-278 (HandleNegativeIndices) / call sites 214,257.
// Pre-fix: an indices element >= the corresponding data dim makes
//   dataIdx = srcShifts[i]*index point far past the source buffer, and
//   cpu_memcpy at gather_nd.cpp:216 reads OOB (ASan: heap-buffer-overflow READ).
// Post-fix: HandleNegativeIndices throws ov::Exception, surfaced by infer().
//
// Harness: ov_cpu_unit_tests (gtest). Build a v8 GatherND model, set a crafted
// out-of-range indices tensor, run on CPU, and assert the OOB index is rejected.
//
// TODO(symbols): confirm includes/helpers against an existing model-driven test in
//   openvino/src/plugins/intel_cpu/tests/unit/ ; the executor is a private nested
//   struct (gather_nd.h:46) so it cannot be unit-tested directly — drive via ov::Core.
// TODO(precision): data is i32 (4 bytes); shape [4,4] -> stride_0=4 elements.
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(GatherND_CPU_OOB, RejectsOutOfRangeIndex) {
    // data: [4,4] i32 ; indices: [1,2] i32 (sliceRank=2, batchDims=0)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4, 4});
    auto idx  = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 2});
    auto gnd  = std::make_shared<op::v8::GatherND>(data, idx, /*batch_dims=*/0);
    auto res  = std::make_shared<op::v0::Result>(gnd);
    auto model = std::make_shared<Model>(ResultVector{res}, ParameterVector{data, idx});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> data_vals(16);
    for (int i = 0; i < 16; ++i) data_vals[i] = i;
    Tensor data_t(element::i32, Shape{4, 4}, data_vals.data());

    // Crafted: row index 1000 is far outside data dim 0 (==4).
    std::vector<int32_t> idx_vals = {1000, 0};
    Tensor idx_t(element::i32, Shape{1, 2}, idx_vals.data());

    req.set_input_tensor(0, data_t);
    req.set_input_tensor(1, idx_t);

    // Pre-fix: ASan heap-buffer-overflow READ inside cpu_memcpy (gather_nd.cpp:216).
    // Post-fix: HandleNegativeIndices throws -> infer() reports an ov::Exception.
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests ; run: ./ov_cpu_unit_tests --gtest_filter=GatherND_CPU_OOB.RejectsOutOfRangeIndex . Pre-fix (ASan build) expected failure: 'ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size N' originating in cpu_memcpy from GatherND::GatherNDExecutor::gatherBlocks (gather_nd.cpp:216). Post-fix: HandleNegativeIndices throws ov::Exception ('GatherND: index out of bounds'), infer() rethrows, ASSERT_ANY_THROW passes. NOTE: skeleton — verify model-driven test includes/helpers against an existing test under intel_cpu/tests/unit/ before use.

## Suggested fix
In `HandleNegativeIndices`, add an upper-bound check after normalization and throw (or clamp with an error) if the index is still out of range: `if (index < 0 || static_cast<size_t>(index) >= srcDims[idx + batchDims]) OPENVINO_THROW("GatherND: index out of bounds");`. Alternatively, add a per-element guard at the call site inside the `for (size_t i = 0; i < sliceRank; i++)` loop in `gatherBlocks` before line 214: `if (index < 0 || static_cast<size_t>(index) >= srcDims[i + batchDims]) OPENVINO_THROW(...);`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #15.
