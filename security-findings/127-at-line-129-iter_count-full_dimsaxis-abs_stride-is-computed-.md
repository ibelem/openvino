# Security finding #127: At line 129, `iter_count = full_dims[axis] / abs_stride` is compute…

**Summary:** At line 129, `iter_count = full_dims[axis] / abs_stride` is compute…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Crash/DoS: a crafted ONNX or OpenVINO IR model with a TensorIterator or Loop op containing a SliceInputDescription or ConcatOutputDescription with stride=0 triggers undefined behavior (integer division by zero) at model-load time during `createPrimitive` → `prepareParamsImpl` → `prepareInputPorts`/`prepareOutputPorts` → `PortIteratorHelper` constructor. This kills the inference engine process immediately.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129` — `PortIteratorHelper::PortIteratorHelper()`
**Validated for repos:** openvino
**Trust boundary:** PortMap.stride sourced from untrusted model graph attributes (SliceInputDescription::m_stride / ConcatOutputDescription::m_stride)

## Description / Root cause
At line 129, `iter_count = full_dims[axis] / abs_stride` is computed where `abs_stride = std::abs(stride)` and `stride` comes directly from `slice_rule.stride`. The stride field is populated at lines 521 and 549 via `static_cast<int>(output_desc->m_stride)` and `static_cast<int>(slice_desc->m_stride)` with no non-zero check anywhere in the call chain. If a crafted model sets stride=0, `abs_stride` is 0 and the integer division at line 129 is a division by zero.

**Validator analysis:** The vuln type CWE-369 (Divide By Zero) is accurate: integer division `full_dims[axis] / abs_stride` at line 129 with abs_stride==0 is UB/crash. Impact (crash/DoS at model prepare time) is accurate — the construction happens in prepareInputPorts/prepareOutputPorts during createPrimitive/prepareParams, and no upstream check rejects stride==0 (core validate_and_infer_types divides only by part_size, lines 98/179, never by stride). The proposed fix is correct but the OPENVINO_ASSERT placement should be BEFORE the division (between lines 128 and 129), not merely 'after computing abs_stride at 126' if that leaves the division unguarded — placing it after line 127 and before 129 is sufficient. Better/defense-in-depth: also reject stride==0 at the supportedPrimitiveDescriptors/initSupportedPrimitiveDescriptors or when populating inputPortMap/outputPortMap (lines 546-552, 518-524) via CPU_NODE_ASSERT, since stride==0 is never a valid Slice/Concat descriptor. Note sign_of_stride logic (line 127) already implies stride should be non-zero, so a single guard `OPENVINO_ASSERT(abs_stride != 0, ...)` before line 129 fixes the immediate crash; the model-load-time check is the more robust fix as it produces a clean error earlier.

## Exploit / Proof of Concept
An attacker crafts an ONNX model with a TensorIterator node containing a `SliceInputDescription` with `m_stride=0`. When this model is loaded by the OpenVINO CPU EP (via `createPrimitive` → `prepareParamsImpl` → `prepareInputPorts`), `PortIteratorHelper` is constructed with `slice_rule.stride=0`, `abs_stride` becomes 0, and the division `full_dims[axis] / abs_stride` at line 129 performs integer division by zero, crashing the host process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129
//   iter_count = full_dims[axis] / abs_stride;  // abs_stride==0 -> int div-by-zero
// stride is taken unchecked from SliceInputDescription::m_stride (line 549) /
// ConcatOutputDescription::m_stride (line 521). Core validate_and_infer_types
// never guards stride!=0 (core/src/op/tensor_iterator.cpp:98 divides by part_size).
//
// This test builds a TensorIterator whose body slices input axis with stride=0
// and runs it on the CPU plugin. Pre-fix: SIGFPE / UBSan integer-divide-by-zero
// during compile_model->createPrimitive->prepareParams->prepareInputPorts.
// Post-fix: the node validation rejects stride==0 with an ov::Exception.
//
// Target test harness: ov_cpu_unit_tests (intel_cpu/tests/unit).
// NOTE: SKELETON — exact builder helpers / include paths must be confirmed by
// reading intel_cpu/tests/unit and core SubGraphOp builder usage before use.

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(TensorIteratorCpu, SliceInputStrideZeroIsRejected) {
    // ---- body: single param -> relu -> result (shape [1, 4]) ----
    auto body_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 4});
    auto body_relu = std::make_shared<op::v0::Relu>(body_param);
    auto body_res = std::make_shared<op::v0::Result>(body_relu);
    auto body = std::make_shared<Model>(ResultVector{body_res}, ParameterVector{body_param});

    // ---- outer TI input [3, 4] sliced on axis 0 ----
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    auto ti = std::make_shared<op::v0::TensorIterator>();
    ti->set_body(body);

    // CRAFTED: stride = 0 (start=0, stride=0, part_size=1, end=-1, axis=0)
    // SliceInputDescription(input_idx, body_param_idx, start, stride, part_size, end, axis)
    ti->set_sliced_input(body_param, data, /*start=*/0, /*stride=*/0,
                         /*part_size=*/1, /*end=*/-1, /*axis=*/0);
    auto out = ti->get_iter_value(body_res, -1);

    auto model = std::make_shared<Model>(OutputVector{out}, ParameterVector{data});

    Core core;
    // Pre-fix this compile (which triggers createPrimitive/prepareParams) crashes
    // via integer divide-by-zero at tensoriterator.cpp:129. Post-fix it must throw.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests; Run: ./ov_cpu_unit_tests --gtest_filter='TensorIteratorCpu.SliceInputStrideZeroIsRejected'. Pre-fix expected failure: SIGFPE / UBSan 'division by zero' (or '-fsanitize=integer-divide-by-zero') inside PortIteratorHelper::PortIteratorHelper at tensoriterator.cpp:129. Post-fix: test passes because compile_model throws ov::Exception from the added stride!=0 validation. TODO: confirm exact set_sliced_input signature/arg order and CPU device availability in the ov_cpu_unit_tests harness before committing.

## Suggested fix
Add a validation check immediately after computing `abs_stride` at line 126: `OPENVINO_ASSERT(abs_stride != 0, "TensorIterator PortMap stride must not be zero");`. Ideally also validate this at model-loading time in `createPrimitive` (around lines 518-552) when populating `inputPortMap`/`outputPortMap` from the op descriptors, e.g., `CPU_NODE_ASSERT(slice_desc->m_stride != 0, "SliceInputDescription stride must be non-zero");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #127.
