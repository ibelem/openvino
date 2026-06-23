# Security finding #149: `stride` is stored as a plain `int` (reorg_yolo.h:33). At line 79, …

**Summary:** `stride` is stored as a plain `int` (reorg_yolo.h:33). At line 79, …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Undefined behaviour (UB) from signed integer overflow. In practice on x86 with optimising compilers, this can produce a negative `stride*stride` that propagates into `srcIndex` at line 94, yielding an out-of-bounds read from `src_data` — a potential information disclosure or controlled crash. Also undermines the `ic_off == 0` safety check even if one is added without casting to a wider type.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:79` — `ReorgYolo::execute()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled model file → stride attribute (stored as `int stride`, line 33 of reorg_yolo.h) → execute() line 79

## Description / Root cause
`stride` is stored as a plain `int` (reorg_yolo.h:33). At line 79, `stride * stride` is a signed 32-bit multiplication with no overflow guard. For stride ≥ 46341, `stride * stride` exceeds INT_MAX (2,147,483,647) — signed integer overflow is undefined behaviour in C++. The resulting wrapped value could be negative or a small positive number, causing `ic_off` to be computed from a nonsensical divisor (e.g., negative → implementation-defined quotient; small → wrong ic_off > 0 that is then used to compute an out-of-bounds `srcIndex` at line 94).

**Validator analysis:** CWE-190 is accurate at line 79: `stride` is a plain `int` (reorg_yolo.h:33) assigned from an unbounded `size_t` stride (reorg_yolo.cpp:52) with no upper-bound validation in the op (core/src/op/reorg_yolo.cpp) or the node. For stride>=46341, `stride*stride` overflows INT_MAX — signed-overflow UB. The static-shape path is effectively unreachable because reorg_yolo_shape_inference.hpp:32-36 forces C>=stride*stride (~2.1e9), an unallocatable tensor; HOWEVER that check is short-circuited when the channel dimension is dynamic (line 35 `input_shape[1].is_dynamic() || ...`). A dynamic-C model with a small runtime IC and large stride is therefore loadable and executable (needPrepareParams()==false, executeDynamicImpl->execute), so line 79 overflows at runtime — the finding is real and reachable from a crafted model file. The stated impact overstates the OOB-read case: the realistic and certain consequence is a divide-by-zero / SIGFPE, because `ic_off = IC/(stride*stride)` collapses to 0 (whether or not the multiply wraps) and line 88 then does `ic % ic_off` (modulo by zero) — this is more precisely CWE-369 layered on the CWE-190. The proposed fix (widen `stride` to int64_t and multiply in 64-bit, plus a constructor range check) removes the UB but is INSUFFICIENT by itself: with a dynamic input shape the runtime IC can still be smaller than stride*stride, yielding ic_off==0 and a divide-by-zero at lines 88-89. The fix must additionally guard before the loop, e.g. `CPU_NODE_ASSERT(ic_off > 0, ...)` computed from a widened `int64_t ic_off = IC / (static_cast<int64_t>(stride)*stride)`, and ideally re-validate C>=stride*stride at execute time for the dynamic-shape path.

## Exploit / Proof of Concept
Supply stride = 50000 in a crafted model. `50000 * 50000 = 2,500,000,000`, which overflows a signed 32-bit int (wraps to a large negative value or triggers UB). `IC / (negative)` gives a negative `ic_off`, or with UB, the compiler may misoptimise and produce arbitrary behaviour. With stride=46341 (smallest overflow), `srcIndex` at line 94 can go negative, yielding a read below `src_data`'s base address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 / CWE-369 in
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:79
//   int ic_off = IC / (stride * stride);   // signed 32-bit overflow for stride>=46341,
//                                          // and ic_off==0 -> modulo-by-zero at line 88
// Pre-fix: building/executing a ReorgYolo with a huge stride (e.g. 50000) on a
//   dynamic-channel input whose concrete C is small triggers UB / SIGFPE (or an
//   out-of-bounds read flagged by ASan) inside ReorgYolo::execute.
// Post-fix: the node must reject the out-of-range stride / zero ic_off and throw
//   ov::Exception instead of executing.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). The exact single-node test
// scaffolding (graph-builder helper, infer-request driver) varies across the
// CPU unit-test tree, so this is emitted as a SKELETON.

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(ReorgYoloCpu, RejectsOverflowingStride) {
    // TODO: confirm the canonical single-op CPU test helper in
    //       src/plugins/intel_cpu/tests/unit/ (e.g. a fixture that compiles a
    //       Model on the "CPU" device and runs infer). Replace the manual
    //       core.compile_model below with that helper if one exists.

    // Dynamic channel dimension so reorg_yolo_shape_inference.hpp:35 skips the
    // C >= stride*stride validation and the model is constructible.
    auto input = std::make_shared<op::v0::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), 8, 8});

    // stride = 50000 -> 50000*50000 overflows int32 at reorg_yolo.cpp:79.
    const size_t kOverflowStride = 50000;
    auto reorg = std::make_shared<op::v0::ReorgYolo>(input, Strides{kOverflowStride, kOverflowStride});
    auto result = std::make_shared<op::v0::Result>(reorg);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // Concrete small C (=4) << stride*stride forces ic_off==0 / overflow at
    // execute time.
    Tensor in(element::f32, Shape{1, 4, 8, 8});
    req.set_input_tensor(in);

    // TODO: pre-fix this either SIGFPEs / ASan-aborts (no clean throw) — confirm
    //       whether the post-fix surfaces ov::Exception from infer(); adjust the
    //       matcher accordingly.
    EXPECT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=ReorgYoloCpu.RejectsOverflowingStride . Pre-fix expectation: UBSan reports 'signed integer overflow: 50000 * 50000 cannot be represented in type int' at reorg_yolo.cpp:79, or a SIGFPE (integer divide/modulo by zero) at line 88; the EXPECT_ANY_THROW fails because the process aborts/crashes instead of throwing. Post-fix expectation: the widened 64-bit multiply + `ic_off>0` / stride range CPU_NODE_ASSERT converts this into an ov::Exception, so infer() throws and the test passes.

## Suggested fix
Change the `stride` member to `int64_t` (or `size_t`) and perform the multiplication in 64-bit: `int64_t ic_off = IC / (static_cast<int64_t>(stride) * stride);`. Add a range check in the constructor: `CPU_NODE_ASSERT(stride > 0 && stride <= 65535, "ReorgYolo: stride out of safe range");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #149.
