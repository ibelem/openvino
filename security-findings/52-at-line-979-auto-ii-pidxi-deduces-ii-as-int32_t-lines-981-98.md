# Security finding #52: At line 979, `auto ii = pidx[i]` deduces `ii` as `int32_t`. Lines 9…

**Summary:** At line 979, `auto ii = pidx[i]` deduces `ii` as `int32_t`. Lines 9…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read of memory before the `psrc` allocation. A crafted ONNX/OV model supplying sufficiently negative indices (e.g. INT32_MIN) in the GATHER_INDICES tensor causes `exec1DCase()` to read arbitrary memory preceding the source data buffer. This can leak sensitive heap data or crash the process (SIGSEGV / access violation), affecting any application running inference on an untrusted model with a 1-D int32 gather of ≤64 elements.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted int32_t index values from the GATHER_INDICES input tensor flowing into exec1DCase() without post-normalization sign validation

## Description / Root cause
At line 979, `auto ii = pidx[i]` deduces `ii` as `int32_t`. Lines 981-982: when `reverseIndexing==true` and `ii < 0`, the code does `ii += axisDim` (where `axisDim` is `size_t`). The compound-assignment promotes `ii` to `size_t`, adds `axisDim`, then truncates back to `int32_t`. If `pidx[i]` is sufficiently negative (e.g. INT32_MIN with `axisDim=64`: `INT32_MIN+64` still fits in `int32_t` as a negative value), `ii` remains negative after normalization. Line 987 then uses this negative `int32_t` directly as `psrc[ii]`, causing pointer arithmetic before the buffer start. There is NO `ii >= 0` guard after the normalization block (lines 981-986). By contrast, `execReference()` at lines 945-947 correctly casts to `size_t` then uses `idx < static_cast<size_t>(axisDim)` to safely reject any still-negative result.

**Validator analysis:** CWE-125 Out-of-bounds Read is accurate. The flaw is real: exec1DCase() (lines 978-988) normalizes only negative indices but performs NO post-normalization bounds check before psrc[ii], unlike execReference() which gates on `idx < axisDim` (line 947). The claimed INT32_MIN truncation path is correct — `ii += axisDim` promotes to size_t then narrows back to a still-negative int32_t (e.g. 0x80000040). Note the defect is actually broader than the report states: even a POSITIVE index >= axisDim (e.g. 50 with axisDim=10) reads OOB, and the non-reverseIndexing branch sets ii=axisDim causing a one-past-end read; the function simply lacks any upper-bound check. Reachable: prepareParams() sets canOptimize1DCase for 1-D i32 tensors <=64 elements and returns without validating index values, and execute() dispatches to exec1DCase(); reverseIndexing defaults to true for v8::Gather. The proposed fix (cast to size_t, guard `idx < axisDim`, else write 0) is correct and sufficient — it mirrors execReference() and simultaneously rejects still-negative wrapped values AND positive out-of-range indices, matching ONNX/OV out-of-range semantics. Recommend applying the identical guard pattern as the validated reference path.

## Exploit / Proof of Concept
Provide a model where the Gather node has a 1-D int32 data tensor of length ≤64 and a 1-D int32 indices tensor of length ≤64, with `reverseIndexing==true` and at least one index value of INT32_MIN (−2147483648). `prepareParams()` sets `canOptimize1DCase=true` and returns immediately without validating index values. During `execute()`, `exec1DCase()` is called. For that index: `ii = INT32_MIN`, `ii < 0` is true, `ii += axisDim` (e.g. 64) yields `INT32_MIN+64` (still negative in int32_t). `psrc[INT32_MIN+64]` computes `psrc + (INT32_MIN+64)` as signed pointer arithmetic, reading ~2 GB before the buffer — either leaking memory contents or faulting.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in Gather::exec1DCase()
// Unchecked sink: openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no `ii >= 0 && ii < axisDim` guard after
//   the negative-index normalization at lines 980-986.
// Pre-fix: an index of INT32_MIN (or any value >= axisDim) on a 1-D i32
//   data tensor of <=64 elements drives exec1DCase() (selected by
//   prepareParams() lines 395-404) to read before/after the psrc buffer
//   -> ASan heap-buffer-overflow / SIGSEGV.
// Post-fix: the `idx < axisDim` guard zero-fills the OOB index, so the
//   graph executes safely and the output element is 0.
//
// HARNESS: ov_cpu_unit_tests / subgraph functional test (gtest).
// SKELETON: exact builder helpers + result-comparison macros must be
//   filled from the nearest existing Gather subgraph test before use.

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm namespace / fixture base used by intel_cpu unit tests
//       (e.g. ov::test::SubgraphBaseTest under intel_cpu/tests/unit).
TEST(GatherCpu1DCaseOobRead, NegativeIndexInt32MinIsRejectedNotOob) {
    using namespace ov;

    // 1-D i32 data of length <= 64 so prepareParams() enables exec1DCase.
    const std::vector<int32_t> dataVals(64, 7);
    auto data = std::make_shared<op::v0::Constant>(element::i32, Shape{64}, dataVals);

    // Indices input carrying the malicious INT32_MIN value.
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto axis = std::make_shared<op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    // v8::Gather -> reverseIndexing == true by default (gather.cpp:115).
    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{indices});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor idxTensor(element::i32, Shape{1});
    idxTensor.data<int32_t>()[0] = std::numeric_limits<int32_t>::min(); // INT32_MIN
    req.set_input_tensor(idxTensor);

    // Pre-fix: ASan reports heap-buffer-overflow inside exec1DCase here.
    // Post-fix: runs cleanly; out-of-range index zero-filled.
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor();
    EXPECT_EQ(out.data<int32_t>()[0], 0); // guard writes 0 for OOB index

    // TODO: verify get_output_tensor() element type matches outPrecision and
    //       adjust dtype accessor if the CPU plugin promotes i32 output.
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (build with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=GatherCpu1DCaseOobRead.* . Expected pre-fix: ASan 'heap-buffer-overflow READ' originating in ov::intel_cpu::node::Gather::exec1DCase (gather.cpp:987) (or SIGSEGV without ASan); post-fix: test passes with output element == 0. NOTE: confirm the correct intel_cpu unit/subgraph target name and Gather test fixture from intel_cpu/tests/unit before running.

## Suggested fix
After the normalization block, add the same guard pattern used in `execReference()`: replace `pdst[i] = psrc[ii];` with:
```cpp
const size_t idx = static_cast<size_t>(ii);
if (idx < axisDim) {
    pdst[i] = psrc[idx];
} else {
    pdst[i] = 0;
}
```
This converts `ii` to `size_t` so any still-negative value wraps to a near-SIZE_MAX value that the `idx < axisDim` guard safely rejects, exactly mirroring the validated path in `execReference()`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #52.
