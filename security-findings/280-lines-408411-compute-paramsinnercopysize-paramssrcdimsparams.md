# Security finding #280: Lines 408–411 compute `params.innerCopySize = (params.srcDims[param…

**Summary:** Lines 408–411 compute `params.innerCopySize = (params.srcDims[param…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound → CWE-787: Out-of-bounds Write
**Severity / Impact:** The wrapped `innerCopySize` (potentially ~SIZE_MAX/2 or larger) is passed directly to `cpu_memcpy` at line 528 as `params.innerCopySize * params.dataSize` (padConstantCommon), at line 574 as `params.innerCopySize` (padConstantZero), and at line 619 (padEdge). This causes cpu_memcpy to read/write gigabytes far beyond the actual allocated source and destination heap buffers, producing a heap out-of-bounds read and write. Depending on the attacker's ability to shape heap layout this is exploitable for DoS (crash) or potentially remote code execution.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408` — `Pad::PadExecutor::innerParamsInitialization()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Dynamic pads_begin/pads_end input tensors from ONNX model: int32_t values read at pad.cpp:275–279 (fillingInParameters) flow into params.attrs.padsBegin/padsEnd without magnitude validation

## Description / Root cause
Lines 408–411 compute `params.innerCopySize = (params.srcDims[params.nDimsForWork] + std::min(padsBegin[nDimsForWork], 0) + std::min(padsEnd[nDimsForWork], 0)) * params.shift`. `srcDims[...]` is `size_t`; the two `std::min(..., 0)` calls return `int32_t` negative values. Under C++ usual arithmetic conversions, each negative `int32_t` is converted to `size_t` before addition (e.g., `-1` → `SIZE_MAX`, `-N` → `SIZE_MAX - N + 1`). If either or both negative pad values exceed `srcDims[nDimsForWork]` in magnitude, the sum wraps to a near-`SIZE_MAX` `size_t`. This huge value is stored in `params.innerCopySize` (declared as `size_t` in pad.h:112) with no bounds check or saturation. There is NO guard on this path: lines 403–404 use `std::max(..., 0)` for the pad-count fields, but the copy-size code path at 408–411 uses `std::min` and is entirely unguarded.

**Validator analysis:** Confirmed real and reachable. The intended computation relies on modular size_t arithmetic to subtract cropped elements (works while the final sum stays >=0), but there is no clamp when the negative pads exceed srcDims at the working axis. Crucially, the OUTPUT shape can remain valid while innerCopySize still wraps, because innerCopySize only adds the negative components via std::min(...,0) whereas the output dim adds the full (begin+end). Example: srcDim=3, padsBegin=-13, padsEnd=+13 -> output dim = 3-13+13 = 3 (valid, memory allocates; passes pad_shape_inference.hpp NODE_VALIDATION since CONSTANT mode skips the REFLECT/SYMMETRIC magnitude checks and pad_dim_diff_lb==0 yields output==arg_shape), yet innerCopySize = (3 + (-13) + 0)*shift = -10*shift, which wraps to ~SIZE_MAX. This wrapped value is handed to cpu_memcpy (pad.cpp:528 padConstantCommon, :574 padConstantZero, :619 padEdge), producing a gigabyte-scale heap OOB read/write. The dstMemPtr->isDefined() assert at pad.cpp:261 does NOT catch this because the output shape is legitimately defined. vulnType (CWE-190 -> CWE-787) and impact (heap OOB R/W, DoS/possible RCE) are accurate. The proposed fix (compute in int64_t and clamp innerCopySize to 0 when <=0) correctly neutralizes the cpu_memcpy OOB, but is INSUFFICIENT alone: innerSrcShift (pad.cpp:407 = max(-padsBegin,0)*shift = 13*shift) and srcODims (pad.cpp:390 = padsBegin+srcDims, also wraps) remain corrupted, so the loops at 509/522/567/607 can still index out of bounds. The robust fix is the suggested upstream validation in fillingInParameters/prepareParams (or shape_infer) rejecting any pad value whose negative magnitude exceeds the corresponding source dimension; the signed-clamp fix should be applied in addition.

## Exploit / Proof of Concept
Craft an ONNX model with a Pad node whose pads_begin and pads_end are dynamic (non-constant) inputs. Set the working-dimension pad values to large negative numbers—e.g., padsBegin[nDimsForWork] = -(srcDim + 10) and padsEnd[nDimsForWork] = -(srcDim + 10)—where srcDim is the source dimension size (e.g., 3). With srcDim=3, padsBegin=-13, padsEnd=-13: std::min(-13,0)=-13; size_t arithmetic: 3 + (SIZE_MAX-12) + (SIZE_MAX-12) wraps to SIZE_MAX-22 (on 64-bit). This giant innerCopySize, multiplied by params.shift and dataSize, is handed to cpu_memcpy, reading/writing far past the tensor's heap allocation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408-411
// (and the missing upstream magnitude validation). Pre-fix: a Pad op in CONSTANT
// mode with dynamic pads_begin=-13, pads_end=+13 on a srcDim=3 axis yields a valid
// output shape (3) but innerCopySize wraps to ~SIZE_MAX, driving cpu_memcpy
// (pad.cpp:528/574/619) far past the heap allocation -> ASan heap-buffer-overflow.
// Post-fix: the over-crop is rejected (throw) OR innerCopySize is clamped to 0 so
// no OOB occurs and inference completes deterministically.
//
// TODO(symbols): confirm the exact single-layer test base class / helpers used in
// openvino/src/plugins/intel_cpu/tests/functional (e.g. SubgraphBaseTest / ov::Model
// builder) — names below are best-effort and must be checked against the tree.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

// Build: data[1,1,3] padded on last axis with DYNAMIC pads inputs begin=-13,end=+13.
TEST(intel_cpu_Pad, NegativeCropExceedingDim_NoHeapOverflow) {
    using namespace ov;
    auto data   = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 3});
    auto pbegin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    auto pend   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    auto pval   = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto pad    = std::make_shared<op::v1::Pad>(data, pbegin, pend, pval, op::PadMode::CONSTANT);
    auto model  = std::make_shared<Model>(OutputVector{pad}, ParameterVector{data, pbegin, pend});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float>   in_data(3, 1.0f);
    std::vector<int32_t> begin{0, 0, -13};
    std::vector<int32_t> end{0, 0, 13};
    req.set_input_tensor(0, Tensor(element::f32, Shape{1,1,3}, in_data.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{3}, begin.data()));
    req.set_input_tensor(2, Tensor(element::i32, Shape{3}, end.data()));

    // Pre-fix this infer triggers an ASan heap-buffer-overflow inside cpu_memcpy.
    // Post-fix either the over-crop is rejected, or it runs without OOB.
    // Accept either a clean throw (validation added) or a clean completion (clamp).
    try {
        req.infer();
        SUCCEED() << "completed without heap overflow (clamp path)";
    } catch (const ov::Exception&) {
        SUCCEED() << "rejected over-crop via validation (preferred fix)";
    }
    // TODO: under ASan, the assertion of interest is simply that the process does
    // NOT abort with heap-buffer-overflow; gtest's death is observed by the harness.
}
```
**Build / run:** Build target: ov_cpu_func_tests (or ov_cpu_unit_tests) compiled with -DENABLE_SANITIZER=ON (ASan). Run: ov_cpu_func_tests --gtest_filter='intel_cpu_Pad.NegativeCropExceedingDim_NoHeapOverflow'. Expected pre-fix: ASan reports 'heap-buffer-overflow WRITE' originating in cpu_memcpy via Pad::PadExecutor::padConstant* (pad.cpp:528/574). Expected post-fix: test passes (either ov::Exception thrown by the new magnitude validation, or inference completes with innerCopySize clamped to 0).

## Suggested fix
Before assigning innerCopySize, compute the value in signed arithmetic and clamp to zero: `int64_t copyElems = static_cast<int64_t>(params.srcDims[params.nDimsForWork]) + static_cast<int64_t>(std::min(params.attrs.padsBegin[params.nDimsForWork], 0)) + static_cast<int64_t>(std::min(params.attrs.padsEnd[params.nDimsForWork], 0)); params.innerCopySize = (copyElems > 0) ? static_cast<size_t>(copyElems) * params.shift : 0;` Also add upstream validation in fillingInParameters (lines 273–280) or prepareParams to reject any pad value whose absolute magnitude exceeds the corresponding source dimension, returning an error rather than producing invalid executor state.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #280.
