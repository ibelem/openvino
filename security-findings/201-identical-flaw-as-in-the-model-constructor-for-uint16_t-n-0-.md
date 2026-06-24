# Security finding #201: Identical flaw as in the model constructor: `for (uint16_t n = 0; n…

**Summary:** Identical flaw as in the model constructor: `for (uint16_t n = 0; n…

**CWE IDs:** CWE-835: Loop with Unreachable Exit Condition (Infinite Loop) / CWE-190: Integer Overflow
**Severity / Impact:** Same as the model constructor: infinite loop leading to unbounded heap allocation and process DoS. The cache-restore code path is also affected, widening the attack surface to include cached model loading scenarios.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/src/plugin/compiled_model.cpp:160` — `CompiledModel::CompiledModel (cache/BinaryInputBuffer constructor)()`
**Validated for repos:** openvino
**Trust boundary:** Externally-supplied ov::num_streams configuration property passed into the cache-restore CompiledModel constructor

## Description / Root cause
Identical flaw as in the model constructor: `for (uint16_t n = 0; n < m_config.get_num_streams(); n++)` at line 160 uses a `uint16_t` counter against an `int32_t` limit. The same absence of validation in options.inl and apply_performance_hints applies. The cache-restore path is reachable when a compiled model is loaded from a cache blob, making it exploitable via a maliciously crafted `ov::num_streams` configuration at model-loading time.

**Validator analysis:** The loop defect is real: in the cache-restore CompiledModel ctor (compiled_model.cpp:160-163), the same uint16_t/int32_t mismatch as the model ctor (line 65) exists. apply_performance_hints (execution_config.cpp:406-433) and the finalize path apply no upper bound to ov::num_streams, so get_num_streams() can return an arbitrary int32_t. With num_streams>=65536, uint16_t n wraps from 65535 to 0 and the comparison stays true forever, allocating Graphs unboundedly — an accurate CWE-835/CWE-190 DoS. However, the finding's distinguishing claim is inaccurate: num_streams in this ctor comes from the `config` argument, NOT from the deserialized cache blob (`ib` only yields params/results/graph, lines 86-159). So a 'maliciously crafted cache blob' cannot set num_streams; the only trigger is the application explicitly configuring num_streams>65535, identical to the model-ctor finding (166) — the cache path adds no new untrusted control. The proposed fix (change `uint16_t n` to `int32_t n` and add an upper-bound validator for ov::num_streams in the GPU options) is correct and sufficient; the validator/clamp is the more important half since it also caps the finite-but-huge allocation when 1024<num_streams<65536. Recommend clamping in apply_performance_hints/options validation rather than only widening the counter.

## Exploit / Proof of Concept
Load a cached model with `core.compile_model(cached_stream, "GPU", {{ov::num_streams(70000)}})`. The BinaryInputBuffer constructor reaches line 160 with `get_num_streams()==70000`; `uint16_t n` wraps at 65535 back to 0 and the condition is true again, infinite loop.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for compiled_model.cpp:160-163 (CWE-835/CWE-190).
// Pre-fix: ExecutionConfig with ov::num_streams(70000) reaches the cache-restore
// (and model) ctor loop `for (uint16_t n = 0; n < get_num_streams(); n++)`, where
// the uint16_t counter wraps at 65535 -> infinite loop / unbounded m_graphs growth.
// Post-fix (int32_t counter + num_streams upper-bound validator): num_streams is
// rejected/clamped, so compile completes with a bounded stream count (or throws).
//
// NOTE: This test cannot directly assert an infinite loop (it would hang). It instead
// pins the *fix's* observable contract: an out-of-range num_streams is rejected.
// TODO: confirm exact target/include paths against intel_gpu/tests/unit/ before use.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

// TODO: if intel_gpu unit tests use a fixture (e.g. a cldnn engine fixture under
// intel_gpu/tests/unit/), inherit from it instead of constructing ov::Core directly.
static std::shared_ptr<Model> make_trivial_model() {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 8});
    auto r = std::make_shared<op::v0::Relu>(p);
    auto res = std::make_shared<op::v0::Result>(r);
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{p});
}

TEST(intel_gpu_num_streams, rejects_out_of_range_num_streams) {
    ov::Core core;
    auto model = make_trivial_model();
    // 70000 > UINT16_MAX: pre-fix this drives the wrapping loop at compiled_model.cpp:160.
    // Post-fix the value must be rejected (validator) — assert it throws rather than hangs.
    // TODO: if the fix clamps instead of throwing, replace EXPECT_THROW with a check that
    //       the compiled model's ov::num_streams property is <= the sane max (e.g. 1024).
    EXPECT_THROW(
        { auto cm = core.compile_model(model, "GPU", {ov::num_streams(70000)}); (void)cm; },
        ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_gpu_unit_tests . Run: ov_gpu_unit_tests --gtest_filter=intel_gpu_num_streams.rejects_out_of_range_num_streams . Pre-fix expectation: test hangs/OOMs (infinite loop allocating Graphs at compiled_model.cpp:160) or, under a watchdog, ASan reports runaway heap growth; post-fix expectation: compile_model rejects num_streams=70000 (ov::Exception) or clamps to the validated max. Requires a GPU device; mark/skip on hosts without one.

## Suggested fix
Same remediation as for line 65: change `uint16_t n` to `int32_t n` at line 160, and add an upper-bound validator for `ov::num_streams` in options.inl to reject values above a sane maximum (e.g. 1024).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #201.
