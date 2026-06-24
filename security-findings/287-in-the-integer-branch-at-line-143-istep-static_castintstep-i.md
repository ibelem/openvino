# Security finding #287: In the integer branch at line 143, `iStep = static_cast<int>(step)`…

**Summary:** In the integer branch at line 143, `iStep = static_cast<int>(step)`…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** An ONNX/OV model with an i32 Range node and `delta=0` crashes the inference process unconditionally — a trivially triggerable DoS against any application loading such a model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143` — `Range::getWorkAmount()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX/OV model-supplied i32 scalar RANGE_DELTA entering CPU EP node execution

## Description / Root cause
In the integer branch at line 143, `iStep = static_cast<int>(step)` is computed from the model-supplied RANGE_DELTA scalar, and then `div_up(abs_span, iStep < 0 ? -iStep : iStep)` is called. When `step` is 0, both `iStep < 0` and `-iStep` evaluate to 0, so `div_up(x, 0)` performs integer division by zero — undefined behaviour that manifests as a process-level SIGFPE / structured exception on all mainstream platforms.

**Validator analysis:** Confirmed real. range.cpp:137 reads model-supplied RANGE_DELTA[0] at execution; for i32 precision (execute() case ov::element::i32 → rangeKernel<int32_t> → getWorkAmount<int32_t>), step==0 makes both ternary operands 0, so div_up(x,0) at line 143 evaluates (a+b-1)/b with b=0 — integer divide-by-zero. div_up's assert(b) (general_utils.h:32) is compiled out under NDEBUG, so release builds hit a hardware SIGFPE/0xC0000094 = unconditional process crash (DoS). vulnType CWE-369 and DoS impact are accurate. Reachability requires delta NOT be const-folded — i.e. delta supplied as a runtime input rather than a folded constant — which an attacker controls trivially, so it is reachable from both the ORT OpenVINO-EP boundary (CPU device) and OpenVINO directly. The proposed fix `if (step == 0) return 0;` prevents the crash but silently returns an empty range for a spec-invalid step; a stricter fix is to CPU_NODE_THROW on step==0 (Range with step 0 is invalid). Note the float branch (line 145) is NOT crash-safe either: fabs(span)/0.0 yields inf and static_cast<size_t>(ceil(inf)) is UB, so the guard should cover both branches — best fix: validate step!=0 once at the top of getWorkAmount and throw.

## Exploit / Proof of Concept
Provide an ONNX model with a v4::Range node whose three inputs are all int32, with `delta` set to the scalar constant 0. `rangeKernel<int32_t>` calls `getWorkAmount<int32_t>`, line 142 sets `iStep=0`, line 143 calls `div_up(abs_span, 0)` which divides by zero.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for CWE-369 divide-by-zero at
// openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143 (Range::getWorkAmount).
// Pre-fix: an i32 Range node with runtime delta==0 reaches div_up(abs_span, 0)
// -> integer SIGFPE / crash. Post-fix (step==0 guarded with a throw): the
// compiled model's infer request rejects step==0 instead of crashing.
//
// Harness: ov_cpu_unit_tests / CPU functional tests (intel_cpu/tests).
// SKELETON: exact builder helper names and the CPU-device infer entry point
// must be confirmed against the existing intel_cpu Range tests before use.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_Range_div_by_zero, i32_runtime_delta_zero_must_not_crash) {
    // Build: Range(start, stop, step) all i32 scalars as runtime Parameters
    // so step is NOT const-folded and the CPU node executes getWorkAmount.
    auto start = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto stop  = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto step  = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto model = std::make_shared<Model>(OutputVector{range},
                                         ParameterVector{start, stop, step});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_start(element::i32, Shape{}); t_start.data<int32_t>()[0] = 0;
    Tensor t_stop (element::i32, Shape{}); t_stop.data<int32_t>()[0]  = 10;
    Tensor t_step (element::i32, Shape{}); t_step.data<int32_t>()[0]  = 0; // delta == 0
    req.set_input_tensor(0, t_start);
    req.set_input_tensor(1, t_stop);
    req.set_input_tensor(2, t_step);

    // TODO: confirm the post-fix behaviour: getWorkAmount should throw on step==0.
    // Pre-fix this line SIGFPEs (test process crashes); post-fix it must throw.
    EXPECT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or the intel_cpu functional-test target that links Core+CPU). Run: ov_cpu_unit_tests --gtest_filter='intel_cpu_Range_div_by_zero.*'. Pre-fix expected failure: the test binary terminates with SIGFPE (integer division by zero) at range.cpp:143 via div_up (general_utils.h:33); under a debug/ASan build the assert(b) at general_utils.h:32 fires first. Post-fix expected: req.infer() throws (EXPECT_ANY_THROW passes) instead of crashing.

## Suggested fix
At the start of the integer branch (before line 141), add: `if (step == 0) return 0;` (or throw an error). This mirrors the fix needed for the float branch.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #287.
