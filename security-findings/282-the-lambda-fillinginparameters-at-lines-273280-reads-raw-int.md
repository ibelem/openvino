# Security finding #282: The lambda `fillingInParameters` at lines 273–280 reads raw int32_t…

**Summary:** The lambda `fillingInParameters` at lines 273–280 reads raw int32_t…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Missing validation is the root cause enabling the signed-to-unsigned wraparound at line 390 and subsequent OOB reads in padConstantCommon/padConstantZero. Any application that loads and runs untrusted ONNX models through the OpenVINO CPU plugin is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:273` — `Pad::PadExecutor::paramsInitialization (lambda fillingInParameters)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX model dynamic pads_begin/pads_end input tensor at port PADS_BEGIN_ID (1) and PADS_END_ID (2) — values cross the trust boundary when the model is loaded and executed with attacker-controlled inputs

## Description / Root cause
The lambda `fillingInParameters` at lines 273–280 reads raw int32_t values from the dynamic pad input tensors (`srcMemory[type]->getDataAs<const int32_t>()`) and stores them directly as `static_cast<int>(ptr[i])` into `params.attrs.padsBegin`/`padsEnd` without any range check. Neither upper nor lower bounds are verified. An attacker who can supply the pads_begin tensor (e.g., via a crafted ONNX model) can inject arbitrary negative or very large values.

**Validator analysis:** The defect is real. The lambda at pad.cpp:273-280 stores attacker-controlled int32 pad values into params.attrs.padsBegin/padsEnd with zero validation. workPartition() at pad.cpp:390 computes srcODims[i] = padsBegin[i] + srcDims[i] in size_t; a sufficiently negative padsBegin makes this wrap to ~SIZE_MAX. The interior/padding guard at lines 509 and 554 (`indexes[j] < padsBegin[j] || (size_t)indexes[j] >= srcODims[j]`) then never excludes any position, so every output position is treated as interior and srcIdx = (indexes[idx]-padsBegin[idx])*srcStrides[idx] (lines 522/567) reads far past srcData -> OOB read. CWE-20 / OOB read is accurate. The reachability nuance: the finding's literal exploit (padsBegin=-(srcDim+1) alone) would make the output dimension negative -> dim::padded() returns inf_bound (dimension_util.hpp:117) -> oversized/failed allocation, throwing before exec. But a refined input (e.g. padsBegin=-(srcDim+5) with padsEnd=+10) keeps the output dimension positive (allocation succeeds) while srcODims still wraps, so the OOB read is genuinely reachable in CONSTANT mode — shape inference for CONSTANT performs no per-pad lower-bound check (only REFLECT/SYMMETRIC are guarded at pad_shape_inference.hpp:97-105). The proposed fix (lower-bound check padsBegin[i] >= -(int32)srcDims[i]) is directionally correct but incomplete: it must apply the same bound to padsEnd[i] and ideally reject any pad combination where padsBegin[i] < -srcDims[i] (to prevent srcODims wrap) regardless of the compensating padsEnd. Better: in fillingInParameters, after reading each value, OPENVINO_THROW if val < -(int32_t)srcDims[i] for both begin and end, and additionally assert srcDims[i] + padsBegin[i] + padsEnd[i] >= 0.

## Exploit / Proof of Concept
Provide a dynamic pads_begin input tensor containing any int32_t value that, when added to the corresponding source dimension, would produce a negative result (e.g., padsBegin[i] = -(srcDims[i]+1)). The value passes through fillingInParameters unchecked, reaches workPartition:390, wraps to near SIZE_MAX in size_t arithmetic, defeats the guard at lines 509/554, and triggers an OOB read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing pad-range validation at
// openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:273-280 (fillingInParameters)
// and the resulting srcODims wrap at pad.cpp:390 / OOB read at pad.cpp:522,567.
//
// Pre-fix: running a CONSTANT-mode Pad whose DYNAMIC pads_begin holds a value
// more negative than -srcDim (with pads_end compensating so the output dim stays
// positive and allocation succeeds) makes srcODims[i] = padsBegin[i] + srcDims[i]
// wrap to ~SIZE_MAX, the interior guard never fires, and exec reads past srcData
// (ASan: heap-buffer-overflow READ in cpu_memcpy inside padConstant*).
// Post-fix: paramsInitialization rejects the out-of-range pad and the infer call
// throws ov::Exception instead.
//
// TODO: confirm exact CPU unit-test target/harness and helper symbols by reading
// openvino/src/plugins/intel_cpu/tests/unit/ (target name is ov_cpu_unit_tests).
// TODO: replace the pseudo-graph construction below with the repo's actual
// node-test fixture (e.g. build an ov::op::v1::Pad with Parameter pads_begin/
// pads_end, compile on CPU, set pads_begin = {-(dim+5)}, pads_end = {+10}).

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_Pad, dynamic_pads_out_of_range_negative_is_rejected) {
    // data: shape [8]
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{8});
    // dynamic pad inputs (1-D, length 1)
    auto pb = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pe = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pad = std::make_shared<op::v1::Pad>(data, pb, pe, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad}, ParameterVector{data, pb, pe});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::f32, Shape{8});
    Tensor t_pb(element::i32, Shape{1});
    Tensor t_pe(element::i32, Shape{1});
    // padsBegin = -(srcDim + 5) = -13, padsEnd = +10  -> output dim = 8-13+10 = 5 (>0, alloc ok)
    // but srcODims = padsBegin + srcDim = -13 + 8 = -5 -> wraps to ~SIZE_MAX pre-fix.
    t_pb.data<int32_t>()[0] = -13;
    t_pe.data<int32_t>()[0] = 10;
    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_pb);
    req.set_input_tensor(2, t_pe);

    // Post-fix: pad value out of [-srcDim, ...] range must be rejected, not OOB-read.
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or ov_cpu_func_tests if run through the OV inference API). Run: ov_cpu_unit_tests --gtest_filter='intel_cpu_Pad.dynamic_pads_out_of_range_negative_is_rejected'. Build with -DENABLE_SANITIZER=ON (ASan). Pre-fix expectation: ASan 'heap-buffer-overflow READ' originating in cpu_memcpy within Pad::PadExecutor::padConstantZero/padConstantCommon (pad.cpp:527/573), reached via srcODims wrap at pad.cpp:390. Post-fix expectation: ov::Exception thrown from paramsInitialization (pad.cpp:273-280) so ASSERT_ANY_THROW passes with no overflow. TODO: confirm exact CPU test fixture/macros from openvino/src/plugins/intel_cpu/tests/unit/.

## Suggested fix
After reading each pad value at line 278, add a range check: the value must satisfy `ptr[i] >= -(int32_t)srcDims[i]` (negative pads cannot exceed the source dimension size) and `ptr[i] <= (int32_t)some_max_dim_limit`. Specifically: `int val = static_cast<int>(ptr[i]); if (val < 0 && static_cast<size_t>(-val) > corresponding_src_dim) OPENVINO_THROW("Pad: padsBegin[i] is out of valid range"); parameter[i] = val;`. Passing `srcDims` into the lambda enables this check inline.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #282.
