# Security finding #283: Line 143 calls `div_up(iSpan < 0 ? -iSpan : iSpan, iStep < 0 ? -iSt…

**Summary:** Line 143 calls `div_up(iSpan < 0 ? -iSpan : iSpan, iStep < 0 ? -iSt…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Integer division by zero (SIGFPE / undefined behaviour) in the CPU EP execution thread. An attacker who can supply model inputs (e.g. via an ONNX model with a constant-zero step or a runtime-provided RANGE_DELTA input) can crash the inference process (DoS). Crash is deterministic and repeatable. Affects all release builds where NDEBUG is defined.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143` — `Range::getWorkAmount()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX/OV model-supplied scalar RANGE_DELTA input entering CPU EP node execution at inference time

## Description / Root cause
Line 143 calls `div_up(iSpan < 0 ? -iSpan : iSpan, iStep < 0 ? -iStep : iStep)` where `iStep` is read directly from the model-supplied RANGE_DELTA tensor (line 137) with no zero-check anywhere between the read and the call. `div_up` in `general_utils.h:32-33` has only `assert(b)` as a guard — a no-op in release builds when NDEBUG is defined — followed immediately by the bare integer division `(a + b - 1) / b` at line 33. When b==0 this is integer division by zero.

**Validator analysis:** Confirmed. The int32 branch at range.cpp:140-143 performs integer division by |iStep| where iStep is the RANGE_DELTA scalar read directly from the model-supplied tensor at line 137. The constructor (lines 45-66) validates only the shapes (scalar count) of START/LIMIT/DELTA, never the delta value, and getWorkAmount/rangeKernel have no zero-check. div_up at general_utils.h:31-33 only does assert(b) before (a+b-1)/b, which is a no-op in release builds (NDEBUG), so b==0 yields a true integer divide-by-zero → SIGFPE/UB. The vulnType CWE-369 is accurate and the impact (deterministic DoS crash of the inference thread in release builds) is correct. Note the float path (line 145) is NOT affected the same way: float division by zero yields inf/NaN, not SIGFPE, though the subsequent static_cast<size_t> is itself UB — the genuine SIGFPE is the i32 path. The proposed fix is correct and sufficient: an explicit `if (iStep == 0) OPENVINO_THROW(...)` before line 143 (or a check on the integer delta) makes the guard active in both debug and release, and hardening div_up's assert to OPENVINO_ASSERT is a good defense-in-depth complement. Recommend the throw be placed in getWorkAmount before the div_up call so both the dynamic-shape sizing and execution are protected; also consider rejecting step==0 in the float path for consistency.

## Exploit / Proof of Concept
Load an ONNX model with a Range node whose RANGE_DELTA input is the integer scalar 0. During `Range::execute()` → `rangeKernel<int32_t>()` (line 113→149) → `getWorkAmount<int>()` (line 152), `*stepPtr` is set to 0 at line 137. At line 143 the expression `iStep < 0 ? -iStep : iStep` evaluates to 0, so `div_up(|iSpan|, 0)` is called. The `assert(b)` at `general_utils.h:32` is stripped by NDEBUG, and `(a + b - 1) / b` becomes `(|iSpan| - 1) / 0`, raising SIGFPE and crashing the process. No upstream check rejects step==0 in `getWorkAmount`, `rangeKernel`, or the constructor.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143 (Range::getWorkAmount, i32 path)
// Pre-fix: a Range op with int32 step==0 reaches div_up(|span|,0) -> integer SIGFPE
//          (assert(b) in general_utils.h:32 is stripped under NDEBUG).
// Post-fix: getWorkAmount must reject step==0 (OPENVINO_THROW), so inference throws
//          ov::Exception instead of crashing the worker thread.
//
// TODO(harness): place under openvino/src/plugins/intel_cpu/tests/unit/ (target ov_cpu_unit_tests)
//   or as a single-layer test under tests/functional (target ov_cpu_func_tests).
// TODO(symbols): confirm exact include paths / fixture base class by reading the
//   neighbouring CPU Range tests before use; symbol names below are best-effort.

#include <gtest/gtest.h>
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// Builds a tiny model: Range(start=0, limit=4, step=<delta>) with i32 scalars.
static std::shared_ptr<Model> make_range_model() {
    auto start = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto limit = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto delta = std::make_shared<opset8::Parameter>(element::i32, Shape{});
    auto range = std::make_shared<opset8::Range>(start, limit, delta, element::i32);
    return std::make_shared<Model>(OutputVector{range},
                                   ParameterVector{start, limit, delta});
}

TEST(CpuRangeNode, ZeroIntStepDoesNotDivideByZero) {
    Core core;
    auto model = make_range_model();
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    int32_t start = 0, limit = 4, step = 0;  // step == 0 triggers div_up(|span|,0)
    Tensor tStart(element::i32, Shape{}, &start);
    Tensor tLimit(element::i32, Shape{}, &limit);
    Tensor tStep (element::i32, Shape{}, &step);
    req.set_input_tensor(0, tStart);
    req.set_input_tensor(1, tLimit);
    req.set_input_tensor(2, tStep);

    // Pre-fix: SIGFPE crash inside Range::getWorkAmount -> div_up.
    // Post-fix: getWorkAmount throws on step==0, surfaced as ov::Exception.
    EXPECT_THROW(req.infer(), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (or ov_cpu_func_tests if placed as a functional single-layer test). Run: ./ov_cpu_unit_tests --gtest_filter=CpuRangeNode.ZeroIntStepDoesNotDivideByZero . Pre-fix on a release/NDEBUG build the process dies with SIGFPE (integer divide-by-zero) inside Range::getWorkAmount->div_up (range.cpp:143 / general_utils.h:33); under a debug build the assert(b) aborts. Post-fix the request throws ov::Exception('Range: step must be non-zero') and the test passes. TODO: confirm the CPU test target name and Range test fixture by reading intel_cpu/tests/ before committing.

## Suggested fix
Add an explicit zero-check for integer step before calling div_up in `getWorkAmount`. For example, insert immediately after line 141: `if (iStep == 0) OPENVINO_THROW("Range: step must be non-zero");` This mirrors OpenVINO's existing OPENVINO_ASSERT pattern used elsewhere in the node and makes the check active in both debug and release builds. Additionally, strengthen `div_up` itself by replacing `assert(b)` with `OPENVINO_ASSERT(b, "div_up: divisor must be non-zero")` so the guard is release-safe.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #283.
