# Security finding #286: In `getWorkAmount` at line 145, when `data_t` is `float`, the code …

**Summary:** In `getWorkAmount` at line 145, when `data_t` is `float`, the code …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound / CWE-789: Memory Allocation with Excessive Size Value / CWE-787: Out-of-bounds Write
**Severity / Impact:** If the UB cast yields a very large value (e.g., UINT64_MAX on x86 SSE when converting infinity), `redefineOutputMemory` attempts to allocate a size_t-max byte buffer, causing an out-of-memory crash (DoS). If the allocator returns a small or stale buffer, the subsequent `dst_data[iwork]` write at line 164 writes far past the buffer boundary (heap overflow, potential RCE). If the cast yields 0, the node silently produces a wrong-sized output, corrupting downstream inference results. Any ONNX model or OpenVINO IR with a zero-valued RANGE_DELTA constant or a dynamic scalar initialized to 0.0f can trigger this without any special privilege.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:145` — `Range::getWorkAmount / Range::rangeKernel()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX/OV model-supplied f32 scalar RANGE_DELTA entering CPU EP node execution

## Description / Root cause
In `getWorkAmount` at line 145, when `data_t` is `float`, the code executes `static_cast<size_t>(std::ceil(std::fabs(span) / std::fabs(step)))` with no prior check that `step != 0.0f`. If `step` is 0.0f, `std::fabs(0.0f)` yields 0.0f, the division produces +infinity (or NaN when span is also 0.0f), `std::ceil(+infinity)` returns +infinity, and `static_cast<size_t>(+infinity)` is undefined behaviour per C++ [conv.fpint] because the value is outside the representable range of `size_t`. The resulting `work_amount_dst` (UB, but on x86 typically 0 or UINT64_MAX) is then passed unchecked to `redefineOutputMemory({work_amount_dst})` at line 155 and used as the iteration bound in the `parallel_nt` write loop at lines 161–165 (`dst_data[iwork] = dst_value`).

**Validator analysis:** The defect is real and reachable. Range::getWorkAmount<float> (range.cpp:145) computes static_cast<size_t>(std::ceil(std::fabs(span)/std::fabs(step))) with no step!=0 guard; with step=0 the division is +inf (span!=0) or NaN (span==0), ceil() preserves it, and the float->size_t conversion is UB ([conv.fpint]) — on x86 CVTTSD2SI yields 0x8000000000000000. The int path (line 143) is worse: div_up(x,0) is an integer divide-by-zero → SIGFPE. Reachability hinges on op validation: range_shape_infer (range_shape_inference.hpp:77-89) only enforces step!=0 when step_allows_zero==false, which is the v0 path; the v4 Range (op->get_output_type, line 124-134) passes step_allows_zero=true, and ONNX Range maps to v4 — so a zero step survives validation. The core shape-inference itself also has the same UB (line 101-103: static_cast<uint32_t>(ceil(fabs(span)/fabs(step)))), so for static models the allocated buffer (32-bit UB cast, e.g. 0x80000000) and the CPU node's work_amount_dst (64-bit UB cast, 0x8000000000000000) diverge → either a multi-GB OOM in redefineOutputMemory (dynamic, line 155) or a heap OOB write in dst_data[iwork] (line 164). vulnType (CWE-190/789/787) and DoS/OOB impact are accurate. The proposedFix (guard !std::isnormal(step)→return 0, plus an upper-bound clamp, plus iStep==0 guard) is correct and sufficient to make the CPU node safe (returning 0 produces a zero-iteration loop, no OOB), but it is incomplete for the whole chain: the same non-finite->integer UB cast in range_shape_inference.hpp:101-103 should also be guarded (clamp non-finite strided to 0 / bound it), and ideally v4 validation should clamp the element count rather than rely on the cast. Note std::isnormal also rejects subnormal steps, which is acceptable here.

## Exploit / Proof of Concept
Craft an ONNX model with a Range node whose `delta` input is a float32 scalar constant with value 0.0f (or +0.0 / -0.0). At execution time, `Range::execute` dispatches to `rangeKernel<float>()` (line 110), which calls `getWorkAmount<float>` (line 152). Line 137 reads 0.0f into `*stepPtr`. Line 145 then evaluates `static_cast<size_t>(std::ceil(std::fabs(span) / 0.0f))`. The division is IEEE-754 ±infinity (or NaN). The cast is UB; on most x86 builds the x87/SSE CVTSS2SI instruction yields 0x8000000000000000 or 0. Either way, `work_amount_dst` reaches line 155 (`redefineOutputMemory`) and line 164 (`dst_data[iwork]`) completely unchecked.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143,145
// (Range::getWorkAmount / rangeKernel) where a zero `step` causes integer
// divide-by-zero (i32) or a UB float->size_t cast of ceil(inf/NaN) (f32),
// leading to OOM/OOB. Pre-fix: SIGFPE (i32) or huge work_amount_dst -> ASan
// heap-buffer-overflow on dst_data[iwork] (range.cpp:164). Post-fix: the node
// either rejects the zero step (throw) or yields a 0-element output safely.
//
// Harness: ov_cpu_unit_tests (intel_gpu/intel_cpu tests live under
// src/plugins/intel_cpu/tests/unit). EXACT target/fixture names are unverified
// — confirm against the existing single-layer/node test tree before use.
// TODO: locate the correct CPU node test fixture; intel_cpu single-node tests
// usually go through ngraph_functions/ov::Model + ov::Core::compile_model on
// the CPU device, NOT a direct Range node ctor (node ctors need a Graph).

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/constant.hpp"

// TODO: verify device availability guard / SKIP if CPU plugin not built.
TEST(CPURangeZeroStep, F32_ZeroStep_NoCrash) {
    using namespace ov;
    // v4 Range with f32 const inputs, delta = 0.0f (the unguarded path).
    auto start = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto stop  = op::v0::Constant::create(element::f32, Shape{}, {10.0f});
    auto step  = op::v0::Constant::create(element::f32, Shape{}, {0.0f}); // <-- trigger
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::f32);
    auto model = std::make_shared<Model>(OutputVector{range}, ParameterVector{});

    Core core;
    // Pre-fix: compile/infer reaches getWorkAmount with step==0 -> UB cast ->
    // huge work_amount_dst -> ASan heap-buffer-overflow at range.cpp:164 (or OOM).
    // Post-fix: either an ov::Exception is thrown (preferred) or a 0-element
    // output is produced without OOB. Accept either safe outcome.
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto compiled = core.compile_model(model, "CPU");
            auto req = compiled.create_infer_request();
            req.infer();
        } catch (const ov::Exception&) { /* acceptable: input rejected */ }
    });
}

TEST(CPURangeZeroStep, I32_ZeroStep_NoDivByZero) {
    using namespace ov;
    auto start = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto stop  = op::v0::Constant::create(element::i32, Shape{}, {10});
    auto step  = op::v0::Constant::create(element::i32, Shape{}, {0}); // div_up(x,0) -> SIGFPE pre-fix
    auto range = std::make_shared<op::v4::Range>(start, stop, step, element::i32);
    auto model = std::make_shared<Model>(OutputVector{range}, ParameterVector{});
    Core core;
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto compiled = core.compile_model(model, "CPU");
            compiled.create_infer_request().infer();
        } catch (const ov::Exception&) { /* acceptable */ }
    });
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (ensure -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=CPURangeZeroStep.* . Expected pre-fix: i32 case raises SIGFPE (integer divide-by-zero at range.cpp:143); f32 case raises ASan heap-buffer-overflow / WRITE at range.cpp:164 (or bad_alloc/OOM from redefineOutputMemory at range.cpp:155). Post-fix the cases pass (ov::Exception or empty output). NOTE: exact target name and whether a direct ov::Model path reaches the CPU Range node must be confirmed against src/plugins/intel_cpu/tests/unit before relying on this test.

## Suggested fix
Before line 145, add a guard that rejects or clamps a zero or non-finite step:
```cpp
if (!std::isnormal(step)) {
    // step == 0, subnormal, inf, or NaN — produces no elements or is invalid
    return 0;
}
```
Additionally, add a sanity upper-bound check in `rangeKernel` before `redefineOutputMemory`, e.g.:
```cpp
constexpr size_t MAX_RANGE_ELEMENTS = 1ULL << 28; // 256 M elements
if (work_amount_dst > MAX_RANGE_ELEMENTS)
    CPU_NODE_THROW("Range output size ", work_amount_dst, " exceeds limit");
```
For the integer path (line 143), add a similar `iStep == 0` check before calling `div_up` to prevent integer divide-by-zero.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #286.
