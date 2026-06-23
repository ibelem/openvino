# Security finding #79: Line 352: `(dstStart / specIndicesSize) % betweenBatchAndAxisSize` …

**Summary:** Line 352: `(dstStart / specIndicesSize) % betweenBatchAndAxisSize` …

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Process crash (SIGFPE) during `createPrimitive` at model load time on static-shape models. Any process using the OpenVINO CPU EP to compile/load a crafted model with a zero-element indices tensor is affected. This can be used as a denial-of-service against an inference server.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:352` — `Gather::createPrimitive()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-supplied indices tensor shapes: `specIndicesSize` (uint64_t) is accumulated from attacker-controlled dimensions in `initSupportedPrimitiveDescriptors` line 182 and `prepareParams` line 441; no zero-guard precedes its use as a divisor at line 352

## Description / Root cause
Line 352: `(dstStart / specIndicesSize) % betweenBatchAndAxisSize` — `specIndicesSize` is used as a divisor with no zero-check. Line 354: `(... / afterAxisSize) % specIndicesSize` and line 359: `/ (specIndicesSize * afterAxisSize)` share the same risk. If the model's indices tensor has zero elements in any post-batchDims dimension (product of dims = 0), `specIndicesSize` is 0, causing an integer division-by-zero at line 352 (and modulo by zero at line 354 when `specIndicesSize == 0`).

**Validator analysis:** Confirmed. In initSupportedPrimitiveDescriptors (line 182) specIndicesSize = product of idxDims[batchDims..end]; an empty/zero post-batchDims dimension makes it 0. For a fully static node taking the JIT path (line 304-307, e.g. afterAxisSize==1), createPrimitive enters the non-dynamic branch at line 337. Even though totalWork becomes 0 (line 339 wpt computation is safe), parallel_nt at line 342 still invokes the lambda for each thread, and line 352 executes `dstStart / specIndicesSize` = `0 / 0` unconditionally → integer division by zero → SIGFPE. Lines 354 (`% specIndicesSize`) and 356/359 (`/(... * specIndicesSize * afterAxisSize)`) share the same hazard; afterAxisSize is also a divisor at 354. No surrounding try/catch converts SIGFPE, and there is no upstream zero-element guard on the indices tensor in this path. vulnType (CWE-369 Divide By Zero) and impact (process crash / DoS at model-load time, reachable on static-shape models) are accurate. The proposedFix `if (specIndicesSize == 0 || afterAxisSize == 0) return;` placed before the parallel_nt block at line 340 is correct and sufficient to eliminate every division at 352/354/356/359 since all divisors are products of those two values; execReference handles the empty-output case. The additional overflow guard on the product at line 356 is a reasonable defensive extra but not required to fix this specific CWE-369.

## Exploit / Proof of Concept
Supply a static-shape ONNX model where the Gather node's indices tensor has shape [N, 0, ...] (zero in any post-batchDims dimension). `specIndicesSize` becomes 0 after `std::accumulate`. `createPrimitive` is called for the non-dynamic path; line 339 computes `totalWork / dataElPerVec` (= 0, safe), `wpt` = dataElPerVec. The parallel lambda runs for thread 0 with `dstStart = 0`, then hits line 352: `0 / 0` — SIGFPE.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 in openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:352
// (also :354/:359). Pre-fix: a static Gather with a zero-element indices tensor makes
// specIndicesSize==0 (accumulate at gather.cpp:182), and createPrimitive's parallel_nt
// lambda performs `dstStart / specIndicesSize` == 0/0 -> SIGFPE.
// Post-fix: the `if (specIndicesSize==0 || afterAxisSize==0) return;` guard skips the JIT
// param precompute and the model compiles without crashing (empty output via execReference).
//
// HARNESS: ov_cpu_unit_tests (intel_cpu). This is a SKELETON: the exact CPU-node test
// helpers/symbols were not read, so fill in the TODOs against the real test tree before use.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm the correct fixture/namespace by reading the nearest existing tests under
//       openvino/src/plugins/intel_cpu/tests/unit/ (e.g. how a Model is compiled for "CPU").
TEST(GatherCpuDivZero, EmptyIndicesDoesNotCrashOnCompile) {
    using namespace ov;

    // data: static, non-trivial after-axis collapses to afterAxisSize==1 so JIT path is taken.
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    // indices: STATIC zero-element tensor -> specIndicesSize == product(idxDims) == 0.
    auto indices = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    Core core;
    // Pre-fix this aborts with SIGFPE inside Gather::createPrimitive (gather.cpp:352).
    // Post-fix it must compile cleanly (no throw, no crash).
    // TODO: if empty-shape Gather is rejected earlier by shape inference on this build,
    //       use a [N,0] indices Constant instead so product==0 while rank>0.
    EXPECT_NO_THROW({ auto compiled = core.compile_model(model, "CPU"); });
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter='GatherCpuDivZero.EmptyIndicesDoesNotCrashOnCompile'. Pre-fix expectation: process terminates with SIGFPE (integer divide-by-zero) inside Gather::createPrimitive at gather.cpp:352 during core.compile_model(...,"CPU"); under a sanitizer build this surfaces as 'integer divide by zero' / FPE. Post-fix: test passes (compile succeeds, no FPE).

## Suggested fix
Add a guard before the parallel_nt block: `if (specIndicesSize == 0 || afterAxisSize == 0) return;` — these represent degenerate zero-work graphs and the JIT path should be skipped (execReference handles the empty-output case gracefully). Additionally guard the product `betweenBatchAndAxisSize * specIndicesSize * afterAxisSize` at line 356 against overflow using a pre-multiplication check.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #79.
