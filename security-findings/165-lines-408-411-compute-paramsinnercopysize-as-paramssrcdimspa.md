# Security finding #165: Lines 408-411 compute `params.innerCopySize` as `(params.srcDims[pa…

**Summary:** Lines 408-411 compute `params.innerCopySize` as `(params.srcDims[pa…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** The wrapped enormous `innerCopySize` is passed directly to `cpu_memcpy` at line 528 (as `innerCopySize * dataSize`) and at line 574 (as `innerCopySize`). This triggers a massive out-of-bounds heap read (from `srcData`) and out-of-bounds heap write (into `dstData`), causing heap memory corruption, process crash (DoS), and potentially arbitrary code execution if an attacker crafts the pad values in a dynamically-shaped ONNX/OpenVINO model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408` — `Pad::PadExecutor::innerParamsInitialization()`
**Validated for repos:** openvino
**Trust boundary:** Dynamic-shape model tensor: `padsBegin`/`padsEnd` are read as raw `int32_t` from model inputs in `paramsInitialization` (lines 273-286) without sign validation; attacker controls these values.

## Description / Root cause
Lines 408-411 compute `params.innerCopySize` as `(params.srcDims[params.nDimsForWork] + std::min(padsBegin[nDimsForWork], 0) + std::min(padsEnd[nDimsForWork], 0)) * params.shift`. `srcDims` is `size_t`; `std::min(..., 0)` returns `int32_t`. Under C++ integral promotions, each negative `int32_t` is implicitly converted to `size_t` before addition (e.g. `-5` becomes `2^64-5`), causing silent wraparound to a huge `size_t`. No check verifies `(int64_t)srcDims[i] + padsBegin[i] + padsEnd[i] >= 0` before this arithmetic.

**Validator analysis:** vulnType CWE-190 is accurate: the bug is the implicit int32_t→size_t conversion before the size_t addition at pad.cpp:409-410, producing a wrapped enormous innerCopySize. The impact (OOB heap read from srcData / OOB write into dstData via cpu_memcpy at lines 528/574) is correct, but the 'arbitrary code execution' framing is speculative — the realistic, demonstrable impact is heap-buffer-overflow/DoS. Reachability is confirmed for the CONSTANT path specifically: negative padsBegin compensated by positive padsEnd keeps the inferred output dimension small/allocatable (dim::padded(3,100)=103) so the bad_alloc that would otherwise abort a purely-negative pad does NOT fire, and execution reaches the memcpy with the wrapped size. The proposed fix (compute copyLen in int64_t and OPENVINO_THROW when <0 before storing innerCopySize) is correct and sufficient for this sink, and it throws inside the PadExecutor constructor (innerParamsInitialization is called from the ctor at pad.cpp:252), so it surfaces cleanly as an ov::Exception during prepareParams. A stronger fix would validate every dimension once (a negative padsBegin[i] whose magnitude exceeds srcDims[i] also corrupts srcODims at pad.cpp:390 and innerSrcShift at pad.cpp:407 for other dims), e.g. reject any i where (int64_t)srcDims[i]+padsBegin[i]+padsEnd[i] < 0 in workPartition/innerParamsInitialization rather than only at nDimsForWork.

## Exploit / Proof of Concept
Craft a dynamic-shape model where `padsBegin[nDimsForWork]` is a large negative value (e.g. -100) while `srcDims[nDimsForWork]` is small (e.g. 3) and `padsEnd` is 0. `std::min(-100, 0)` returns -100 (int32_t); converted to size_t before addition to 3 → `3 + (2^64-100) + 0 = 2^64-97` (wraps). Multiplied by `params.shift` (also size_t, ≥1) produces another huge size_t stored in `innerCopySize`. On the next `exec()` call, `cpu_memcpy` at line 574 copies `2^64-97` bytes from `srcData`, reading and writing far beyond the allocated buffers.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190 integer overflow in
//   openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:408-411
//   (Pad::PadExecutor::innerParamsInitialization)
// Untrusted pad values enter at pad.cpp:282-287 (dynamic case) with no sign check.
// With CONSTANT mode, padsBegin=-100 and padsEnd=+200 on a source dim of 3:
//   inferred output dim = padded(3, +100) = 103   (allocatable -> no bad_alloc),
//   innerCopySize = (3 + min(-100,0) + min(200,0)) * shift = (3-100)*shift -> size_t wraparound.
// Pre-fix: cpu_memcpy at pad.cpp:528/574 copies ~2^64 bytes -> ASan heap-buffer-overflow.
// Post-fix: innerParamsInitialization computes copyLen in int64_t and OPENVINO_THROW on copyLen<0,
//           so inference is rejected with ov::Exception instead of corrupting the heap.
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit/.
// SKELETON: exact CPU single-layer-test fixture symbols must be confirmed against the
//           intel_cpu/tests/unit tree before use.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"

using namespace ov;

// TODO: confirm the intel_cpu unit-test fixture base / helper (e.g. the test
// helpers under src/plugins/intel_cpu/tests/unit/) and reuse it instead of raw ov::Core.
TEST(CpuPadNegativePadOverflow, InnerCopySizeWraparoundRejected) {
    // Data input: dynamic rank-1 shape so pads are data-dependent (dynamic path).
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto pads_begin = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pads_end   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    // v1 Pad, CONSTANT mode (no begin/end-vs-dim validation in shape_infer for CONSTANT).
    auto pad = std::make_shared<op::v1::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad},
                                         ParameterVector{data, pads_begin, pads_end});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // srcDim = 3, padsBegin = -100, padsEnd = +200  -> output dim 103, innerCopySize wraps.
    Tensor t_data(element::f32, Shape{3});
    std::fill_n(t_data.data<float>(), 3, 0.0f);
    Tensor t_begin(element::i32, Shape{1}); t_begin.data<int32_t>()[0] = -100;
    Tensor t_end(element::i32, Shape{1});   t_end.data<int32_t>()[0]   =  200;
    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_begin);
    req.set_input_tensor(2, t_end);

    // Pre-fix: ASan reports heap-buffer-overflow inside cpu_memcpy (pad.cpp:528/574).
    // Post-fix: PadExecutor ctor throws ov::Exception from innerParamsInitialization.
    EXPECT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter='CpuPadNegativePadOverflow.InnerCopySizeWraparoundRejected'. Pre-fix expected: AddressSanitizer 'heap-buffer-overflow READ/WRITE' originating from cpu_memcpy via Pad::PadExecutor::padConstantZero/padConstantCommon (pad.cpp:528 / pad.cpp:574). Post-fix expected: test passes because innerParamsInitialization throws ov::Exception (copyLen<0) and EXPECT_ANY_THROW catches it. NOTE: confirm the intel_cpu unit-test fixture/helpers under src/plugins/intel_cpu/tests/unit/ before merging; this is an UNVERIFIED skeleton.

## Suggested fix
Before computing `innerCopySize`, perform the arithmetic in `int64_t` and assert/check the result is non-negative:
```cpp
int64_t copyLen = static_cast<int64_t>(params.srcDims[params.nDimsForWork])
    + std::min(params.attrs.padsBegin[params.nDimsForWork], 0)
    + std::min(params.attrs.padsEnd[params.nDimsForWork], 0);
if (copyLen < 0)
    OPENVINO_THROW("Pad: negative innerCopySize for dimension ", params.nDimsForWork);
params.innerCopySize = static_cast<size_t>(copyLen) * params.shift;
```
This prevents the unsigned wraparound and rejects invalid (attacker-supplied) pad values before they reach the `cpu_memcpy` sinks.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #165.
