# Security finding #148: The constructor only asserts `!strides.empty()` (line 51) but never…

**Summary:** The constructor only asserts `!strides.empty()` (line 51) but never…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Reliable process crash (SIGFPE) on any inference call against a model supplying stride=0 for a ReorgYolo node. Denial-of-service against any OpenVINO-based inference service accepting external models.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:79` — `ReorgYolo::execute()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled model file → ov::op::v0::ReorgYolo stride attribute → ReorgYolo::ReorgYolo() (line 52) → ReorgYolo::execute()

## Description / Root cause
The constructor only asserts `!strides.empty()` (line 51) but never checks `stride > 0`. If the model sets stride=0, line 79 evaluates `IC / (0 * 0)` = `IC / 0` — integer division by zero at the point of computing `ic_off`, crashing immediately before the inner loop. No try/catch, no prior check.

**Validator analysis:** The CWE-369 Divide-By-Zero classification is accurate. `stride` is a plain `int` member (reorg_yolo.h:33) set unconditionally from `strides[0]` (reorg_yolo.cpp:52) with no positivity check — only `!strides.empty()` is asserted (line 51). I confirmed reachability: a model with stride=0 survives core graph validation because v0::ReorgYolo::validate_and_infer_types (reorg_yolo.cpp:21) calls shape_infer, whose only division uses Dimension::operator/ which asserts `divisor >= 0` (dimension.cpp:126) — 0 satisfies `>=0`, and the subsequent double-precision ceil/floor(x/0) yields inf rather than a trap, so no exception is raised upstream. Thus execution reaches reorg_yolo.cpp:79 where `IC / (0*0)` is integer division by zero → SIGFPE. There is no surrounding try/catch in execute(). Impact (reliable DoS/SIGFPE on inference of an attacker-supplied model) is accurate. The proposed fix is correct in intent but should be placed AFTER line 52 (where `stride` is assigned), e.g. `CPU_NODE_ASSERT(stride > 0, "ReorgYolo: stride must be positive, got ", stride);`. A more robust fix would also tighten the core op's validate_and_infer_types (reorg_yolo.cpp:21) to reject stride==0 so every backend is protected, and harden Dimension::operator/ to reject divisor==0 (the `>= 0` assert is itself a latent divide-by-zero gap). The CPU-node assertion alone is sufficient to stop the crash on the CPU plugin path.

## Exploit / Proof of Concept
Provide a model with a ReorgYolo operator whose stride attribute is 0. The constructor sets `this->stride = 0`. On the first call to execute(), line 79 computes `IC / (0 * 0)` = division by zero → SIGFPE.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero in ReorgYolo.
// Encodes the fix cited at:
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:52 (constructor: missing stride>0 check)
//   triggering the SIGFPE at reorg_yolo.cpp:79 `IC / (stride*stride)`.
// Pre-fix: compiling/running a ReorgYolo with stride=0 on the CPU plugin reaches
//          `IC / 0` -> SIGFPE (or, with the fix in the core op, an upstream throw).
// Post-fix: the CPU node constructor's CPU_NODE_ASSERT(stride>0, ...) makes
//          compile_model on the CPU device throw ov::Exception before execution.
//
// NOTE: ReorgYolo::execute needs allocated edges/memory, so the cleanest harness
// is to build a tiny ov::Model and compile it on the CPU plugin and assert the
// throw. Symbols/headers below were modelled on the intel_cpu unit test tree but
// MUST be reviewed for exact include paths and target wiring.
//
// TODO: confirm test file lands under src/plugins/intel_cpu/tests/unit/nodes/
//       and is picked up by the ov_cpu_unit_tests glob (see tests/unit/CMakeLists.txt).
#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/runtime/core.hpp"

// stride=0 must be rejected (post-fix) rather than crash with SIGFPE (pre-fix).
TEST(ReorgYoloNodeTest, StrideZeroIsRejectedOnCpu) {
    using namespace ov;

    // [N, C, H, W] with C divisible setup is irrelevant once stride==0.
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 4, 6, 6});

    // stride attribute = 0 (attacker-controlled in a crafted model).
    // TODO: verify the Strides ctor overload; op::v0::ReorgYolo(input, Strides{0,0}).
    auto reorg = std::make_shared<op::v0::ReorgYolo>(param->output(0), Strides{0, 0});
    auto result = std::make_shared<op::v0::Result>(reorg->output(0));
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    Core core;
    // With the constructor fix, CPU node creation throws during compile_model.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
```
**Build / run:** Build: cmake --build build --target ov_cpu_unit_tests. Run: ./ov_cpu_unit_tests --gtest_filter=ReorgYoloNodeTest.StrideZeroIsRejectedOnCpu . Pre-fix expectation: SIGFPE (integer divide-by-zero) at reorg_yolo.cpp:79 during inference, or under ASan/UBSan a 'division by zero' diagnostic; the EXPECT_THROW fails because no ov::Exception is raised before the crash. Post-fix expectation: CPU_NODE_ASSERT(stride>0) throws ov::Exception at compile_model and the test passes. NOTE: skeleton — review include paths, the Strides ctor overload, and that execute() is actually invoked if compile_model alone does not construct the node eagerly (may need core.compile_model(...).create_infer_request().infer()).

## Suggested fix
Add `CPU_NODE_ASSERT(stride > 0, "ReorgYolo: stride must be positive, got ", stride);` immediately after line 52 in the constructor.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #148.
