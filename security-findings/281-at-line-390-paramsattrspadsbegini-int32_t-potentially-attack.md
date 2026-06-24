# Security finding #281: At line 390, `params.attrs.padsBegin[i]` (int32_t, potentially atta…

**Summary:** At line 390, `params.attrs.padsBegin[i]` (int32_t, potentially atta…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error
**Severity / Impact:** The wrapped-huge srcODims[j] (≈SIZE_MAX) defeats the upper-bound guard at padConstantCommon:509 and padConstantZero:554 (`static_cast<size_t>(indexes[j]) >= params.srcODims[j]` is never true). Code falls through to the srcIdx computation at lines 522/567: `srcIdx += (indexes[idx] - params.attrs.padsBegin[idx]) * params.srcStrides[idx]`. Since padsBegin[idx] is negative, the subtraction of a negative produces a large positive offset that is multiplied by srcStrides, resulting in an out-of-bounds srcIdx. This causes an out-of-bounds read from the source tensor buffer, leading to a crash (DoS) or an information leak (reading adjacent memory). Reachable whenever a crafted ONNX model uses dynamic (non-constant) pads_begin with a negative value.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:390` — `Pad::PadExecutor::workPartition()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX model dynamic pads_begin input tensor (port PADS_BEGIN_ID=1) → int32_t buffer read in fillingInParameters without lower-bound validation → stored in params.attrs.padsBegin

## Description / Root cause
At line 390, `params.attrs.padsBegin[i]` (int32_t, potentially attacker-controlled negative value) is added to `params.srcDims[i]` (size_t). The implicit signed-to-unsigned integer promotion means a negative padsBegin[i] wraps to a value near SIZE_MAX in the result, which is stored in `params.srcODims` (VectorDims = std::vector<size_t>). No lower-bound validation exists in fillingInParameters (lines 275–278) or anywhere on the path from trust boundary to this computation.

**Validator analysis:** The signed-to-unsigned defect at pad.cpp:390 is real: padsBegin is std::vector<int32_t> (pad.h:43,50) read verbatim from the dynamic PADS_BEGIN input (pad.cpp:275-278) with no lower-bound validation at node construction (pad.cpp:89-104, dynamic case does nothing) nor in paramsInitialization. The researcher's two concrete exploit scenarios are individually flawed: padsBegin=-1 actually yields the CORRECT crop (modular: (size_t)(-1)+5==4), and padsBegin=-srcDims-100 with padsEnd=0 makes the *output* dim negative, which in StaticShape inference (dimension_util.hpp:114-118, size_t arithmetic never satisfies `<0`) wraps the output dim huge and triggers a bad_alloc before the executor runs. HOWEVER the underlying mechanism is genuinely reachable via a compensated crop: padsBegin[i]=-100, padsEnd[i]=+100 on a dim of 5 gives a valid positive output dim (=5) so allocation succeeds, yet srcODims[i]=padsBegin+srcDims=-95 wraps to ~SIZE_MAX. The upper-bound guard `static_cast<size_t>(indexes[j])>=srcODims[j]` (pad.cpp:509/554) is then never true and the lower guard `indexes[j]<padsBegin[j]` (0<-100) is false, so control falls through to srcIdx+=(indexes[idx]-padsBegin[idx])*srcStrides[idx] (pad.cpp:522/567) producing (indexes+100)*stride — far past the source buffer — an OOB read (DoS/info-leak). So vulnType CWE-195 and impact (OOB read) are accurate. The proposed fix (reject padsBegin[i]<0 && (size_t)(-padsBegin[i])>srcDims[i], or compute srcODims in int64 with an assert) is correct and sufficient because cropping more elements than exist is never legitimate regardless of compensating end padding; it should be applied to padsEnd symmetrically and validated against the actual access range.

## Exploit / Proof of Concept
Supply an ONNX Pad node where pads_begin is a dynamic (non-Constant) input tensor and at inference time provide a value such as -1 for dimension i. At paramsInitialization line 275, ptr[i]=-1 is read as int32_t and stored without validation. At workPartition line 390, `(int32_t)(-1) + (size_t)srcDims[i]` promotes -1 to size_t giving SIZE_MAX, stored in srcODims[i]. At padConstantCommon line 509, `indexes[j]` iterates 0..dstDims[j]-1 which is always < SIZE_MAX, so the guard never triggers. At line 522, `srcIdx += (0 - (-1)) * srcStrides[idx]` = `1 * srcStrides[idx]`, which may be within bounds for srcDims[i]>1, but for large negative values (e.g. padsBegin[i]=-srcDims[i]-100), srcIdx is pushed well beyond the allocated buffer, causing an OOB read of `srcData[srcIdx]`.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195 at openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:390
// (srcODims[i] = padsBegin[i] + srcDims[i] with signed->unsigned wrap on negative padsBegin).
// Pre-fix: a Pad with dynamic pads_begin = -100 and pads_end = +100 on a dim of size 5
// yields a valid positive output dim but srcODims wraps to ~SIZE_MAX, so the bounds guard at
// pad.cpp:509/554 is bypassed and srcIdx at pad.cpp:522/567 reads out of bounds (ASan heap-buffer-overflow).
// Post-fix: paramsInitialization/workPartition must reject pads_begin < -srcDims (OPENVINO_THROW / ov::Exception).
//
// NOTE (skeleton): exact harness target/symbols for intel_cpu single-layer tests are unverified.
// Build under the CPU functional/unit test tree (e.g. ov_cpu_func_tests / ov_cpu_unit_tests).

#include <gtest/gtest.h>
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(intel_cpu_Pad_NegativeCrop, dynamic_pads_begin_oob_is_rejected) {
    // TODO: confirm the canonical CPU single-layer test fixture (see
    // src/plugins/intel_cpu/tests/functional/.../single_layer_tests/pad.cpp) and reuse it instead.
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{5});
    // dynamic (Parameter) pads => shapeHasDataDependency == true, no construction-time validation.
    auto pads_begin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pads_end   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pad_value  = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto pad = std::make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad},
                                         ParameterVector{data, pads_begin, pads_end});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::f32, Shape{5});
    std::fill_n(t_data.data<float>(), 5, 1.0f);
    Tensor t_begin(element::i32, Shape{1});
    Tensor t_end(element::i32, Shape{1});
    t_begin.data<int32_t>()[0] = -100; // crop far beyond dim -> srcODims wraps pre-fix
    t_end.data<int32_t>()[0]   = 100;  // compensate so output dim stays positive (=5), allocation succeeds

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_begin);
    req.set_input_tensor(2, t_end);

    // Pre-fix: ASan reports heap-buffer-overflow inside Pad::PadExecutor::padConstant* (OOB src read).
    // Post-fix: the negative crop exceeding the dim must be rejected with an ov::Exception.
    EXPECT_THROW(req.infer(), ov::Exception);
}
```
**Build / run:** Build ov_cpu_func_tests (or ov_cpu_unit_tests) and run: ov_cpu_func_tests --gtest_filter='intel_cpu_Pad_NegativeCrop.*'. With AddressSanitizer enabled, the pre-fix binary reports 'heap-buffer-overflow READ' in Pad::PadExecutor::padConstant*/padConstantZero (pad.cpp:527/573 cpu_memcpy from srcData[srcIdx]); the post-fix binary instead throws ov::Exception ('Pad: negative padsBegin exceeds srcDim') and the test passes.

## Suggested fix
In `fillingInParameters` (lines 273–280), after storing each value, validate that it is within the legal range: `if (parameter[i] < 0 && (size_t)(-parameter[i]) > srcDims[i]) OPENVINO_THROW("Pad: padsBegin value out of range");`. Alternatively, at the start of `workPartition`, before line 390, add: `for (size_t i = 0; i < params.srcDims.size(); ++i) { OPENVINO_ASSERT(params.attrs.padsBegin[i] >= 0 || (size_t)(-params.attrs.padsBegin[i]) <= params.srcDims[i], "Pad: negative padsBegin exceeds srcDim"); }`. This ensures the `padsBegin[i] + srcDims[i]` expression is non-negative before the unsigned arithmetic is performed. Also consider casting to int64_t for the srcODims computation: `int64_t val = (int64_t)params.attrs.padsBegin[i] + (int64_t)params.srcDims[i]; OPENVINO_ASSERT(val >= 0); params.srcODims.push_back((size_t)val);`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #281.
