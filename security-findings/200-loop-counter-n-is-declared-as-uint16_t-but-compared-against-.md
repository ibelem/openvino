# Security finding #200: Loop counter `n` is declared as `uint16_t` but compared against `m_…

**Summary:** Loop counter `n` is declared as `uint16_t` but compared against `m_…

**CWE IDs:** CWE-835: Loop with Unreachable Exit Condition (Infinite Loop) / CWE-190: Integer Overflow
**Severity / Impact:** An attacker (or misconfigured caller) supplying `ov::num_streams` > 65535 causes an infinite loop inside the CompiledModel constructor, spinning the caller's thread indefinitely and continuously allocating Graph objects until the process OOMs. Affects any application that forwards user-controlled configuration to OpenVINO GPU plugin compilation.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/src/plugin/compiled_model.cpp:65` — `CompiledModel::CompiledModel (model constructor)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Externally-supplied ov::num_streams configuration property passed into CompiledModel constructors

## Description / Root cause
Loop counter `n` is declared as `uint16_t` but compared against `m_config.get_num_streams()` which returns `int32_t`. When `get_num_streams()` returns a value > 65535, `n` overflows back to 0 after reaching 65535 (unsigned wraparound), and the condition `(uint16_t)0 < large_int32` is true again — the loop never terminates. No upper-bound validator exists on `ov::num_streams` in options.inl:9 (no 5th-argument validator lambda, unlike e.g. `inference_precision` at line 11), and `apply_performance_hints` (execution_config.cpp:406–433) only adjusts `m_num_streams` for AUTO/THROUGHPUT/LATENCY hints or exclusive_async_requests — a directly user-supplied value > 65535 is passed through unchanged.

**Validator analysis:** Confirmed: at compiled_model.cpp:65 and :160 the loop counter is uint16_t while get_num_streams() returns int32_t. uint16_t promotes to int for the comparison, but n++ wraps modulo 65536, so for any num_streams>65535 n cycles 0..65535 and the condition stays true forever, with m_graphs.push_back creating Graph objects until OOM. options.inl:9 declares num_streams with NO validator lambda (contrast inference_precision at :11), and apply_performance_hints (execution_config.cpp:406-433) only overrides m_num_streams for AUTO/THROUGHPUT/LATENCY/exclusive_async_requests — a directly user-supplied 100000 passes through unchanged. The vulnType is accurate (CWE-835 driven by CWE-190 unsigned wraparound). Impact (CPU spin + unbounded memory growth/DoS) is accurate. The proposedFix is correct and sufficient: changing the counter to int32_t removes the wraparound at both lines, and an upper-bound validator in options.inl rejects absurd values at the trust boundary (returning AUTO sentinel or 1..N). Both should be applied; the type fix alone is the minimal correctness fix, the validator is defense-in-depth. Reachability for openvinoEp rests on the EP forwarding the num_streams provider option to compile_model, which is its standard pass-through behavior; the defect itself lives in openvino.

## Exploit / Proof of Concept
Call `core.compile_model(model, "GPU", {{ov::num_streams(100000)}})`. `apply_performance_hints` does not clamp this value. The model-constructor loop `for (uint16_t n = 0; n < 100000; n++)` increments n through 0..65535, then n wraps to 0, and 0 < 100000 is true — the loop re-enters and pushes graphs forever, causing unbounded memory growth and DoS.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for compiled_model.cpp:65/:160 (uint16_t loop counter vs int32_t
// get_num_streams()). Pre-fix: ov::num_streams > 65535 wraps the uint16_t counter and
// the CompiledModel ctor loops forever (hang / OOM). Post-fix: either the options.inl
// validator rejects the out-of-range value (throws) or the int32_t counter terminates.
// This test asserts the call returns/throws rather than hanging.
//
// Target: ov_gpu_unit_tests (intel_gpu/tests/unit). Requires a GPU device.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

TEST(gpu_compiled_model, num_streams_overflow_is_bounded) {
    ov::Core core;
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(p);
    auto r = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p});

    // Pre-fix this hangs (uint16_t counter wraps at 65535); post-fix the validator
    // rejects the value at the config boundary -> throws ov::Exception.
    EXPECT_THROW(core.compile_model(model, "GPU", {ov::num_streams(100000)}), ov::Exception);
}
```
**Build / run:** Build target: ov_gpu_unit_tests. Run: ov_gpu_unit_tests --gtest_filter=gpu_compiled_model.num_streams_overflow_is_bounded . Pre-fix on a host with a GPU device the test HANGS in the CompiledModel ctor loop (compiled_model.cpp:65) due to uint16_t wraparound (no sanitizer abort — it is an infinite loop / OOM). Post-fix (options.inl num_streams validator + int32_t counter) the call throws ov::Exception and the EXPECT_THROW passes.

## Suggested fix
1) Add an upper-bound validator in options.inl for `ov::num_streams`, e.g.: `OV_CONFIG_RELEASE_OPTION(ov, num_streams, 1, "...", [](int32_t v){ return v == ov::streams::AUTO || (v >= 1 && v <= 1024); })`. 2) Change the loop counter type from `uint16_t` to `int32_t` (matching the type returned by `get_num_streams()`) in compiled_model.cpp lines 65 and 160: `for (int32_t n = 0; n < m_config.get_num_streams(); n++)`. Both fixes are needed: the type fix prevents silent counter overflow, and the validator rejects out-of-range values at the trust boundary.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #200.
