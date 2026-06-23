# Security finding #60: At line 205-206, `scale_group_size` is computed as `getInputShapeAt…

**Summary:** At line 205-206, `scale_group_size` is computed as `getInputShapeAt…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Integer division by zero causes SIGFPE (crash / DoS) on x86 when the CPU plugin attempts to initialise a GatherCompressed node whose scale or zero-point input has a static shape containing a zero dimension (e.g. [N, 0]). This is reachable whenever an attacker-controlled model is loaded and compiled, i.e., at model compilation time before any inference occurs.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvino
**Trust boundary:** Externally supplied model graph → GatherCompressed node GATHER_SCALE / GATHER_ZP input shapes → CPU plugin node initialisation (initSupportedPrimitiveDescriptors)

## Description / Root cause
At line 205-206, `scale_group_size` is computed as `getInputShapeAtPort(GATHER_DATA).getElementsCount() / getInputShapeAtPort(GATHER_SCALE).getElementsCount()`. At line 217-218, `zp_group_size` is computed similarly using GATHER_ZP. `Shape::getElementsCount()` (cpu_shape.h:165-173) returns the product of all `minDims`; if any dimension is 0 the product is 0. No `hasZeroDims()` check (cpu_shape.h:157) is performed on the GATHER_SCALE or GATHER_ZP shapes before either division, so either denominator can be zero.

**Validator analysis:** The flaw is real at the code level: in the `compressed` branch (entered at gather.cpp:191 when the GatherCompressed node has 4 or 5 inputs), lines 206 and 218 perform `getElementsCount() / getElementsCount()` on the scale/zp inputs. `Shape::getElementsCount()` (cpu_shape.h:165-173) multiplies all `minDims` and returns 0 if any dimension is 0; there is no `hasZeroDims()` (cpu_shape.h:157) guard, so a static scale/zp shape such as [1024,0] makes the denominator 0 → SIGFPE. Note `getElementsCount()` asserts the shape is Static (cpu_shape.h:166), so a *dynamic* scale shape throws an ov exception rather than crashing — the SIGFPE only occurs for a static zero-dim shape, which is a legal degenerate tensor in OpenVINO. The vulnType CWE-369 and the DoS impact at model-compile time are accurate; this triggers in initSupportedPrimitiveDescriptors before inference. The proposed fix is correct and sufficient: a `CPU_NODE_ASSERT(scale_count != 0, ...)` (and the zp counterpart) converts the crash into a normal node-validation exception, consistent with the other CPU_NODE_ASSERT checks already in this file (e.g. lines 95-96, 106). The `hasZeroDims()` alternative is equally valid. For openvinoEp the defect is non-existent in its own sources (it merely hands the model to OpenVINO for compilation), so `na` is correct rather than a pass-through `validated`.

## Exploit / Proof of Concept
Craft an ONNX/IR model that creates a GatherCompressed operation (4 or 5 inputs) where the GATHER_SCALE input has a static shape with at least one zero dimension, e.g. shape [1024, 0]. During `Node::init` → `Gather::initSupportedPrimitiveDescriptors`, the `compressed` branch is entered (line 191), `getInputShapeAtPort(GATHER_SCALE).getElementsCount()` returns 0 (product over [1024,0]), and the division at line 206 performs integer division by zero, triggering SIGFPE and crashing the inference engine process. The same applies to the GATHER_ZP path at line 218 when 5 inputs are present.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205-206 (GATHER_SCALE)
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:217-218 (GATHER_ZP)
// where scale_group_size / zp_group_size are computed as
//   GATHER_DATA.getElementsCount() / GATHER_SCALE(or ZP).getElementsCount()
// with no zero check. A GatherCompressed node whose scale/zp input has a static
// zero-dim shape (e.g. [1024, 0]) makes the denominator 0 -> SIGFPE during
// Gather::initSupportedPrimitiveDescriptors (i.e. at compile time).
//
// This test builds a GatherCompressed (ov::op::internal::GatherCompressed) graph
// with a zero-dim scale constant and compiles it on CPU. PRE-FIX: the process
// dies with SIGFPE (integer division by zero) inside the CPU Gather node.
// POST-FIX: Gather::initSupportedPrimitiveDescriptors throws via CPU_NODE_ASSERT,
// surfaced as an ov::Exception by core.compile_model, which we assert below.
//
// Harness: ov_cpu_unit_tests (the intel_cpu component test target).
// NOTE: SKELETON — exact GatherCompressed factory signature and the test helper
// includes must be confirmed against intel_cpu/tests/unit before use.

#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

// TODO: confirm the correct internal op header path for GatherCompressed in this
//       tree (search intel_cpu transformations for ov::op::internal::GatherCompressed).
// #include "transformations/op_conversions/.../gather_compressed.hpp"  // TODO: real path

using namespace ov;

TEST(CpuGatherCompressed, ZeroDimScaleShapeIsRejectedNotSIGFPE) {
    // data: u8 [1024, 16]
    auto data = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{1024, 16});
    // indices: i32 [4]
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    // axis const = 0
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    // scale const with a ZERO dimension -> getElementsCount() == 0 (the divisor)
    auto scale = op::v0::Constant::create(element::f32, Shape{1024, 0}, std::vector<float>{});

    // TODO: construct ov::op::internal::GatherCompressed with the exact ctor
    //       (data, indices, axis, batch_dims, scale[, zp]). Confirm arg order.
    // auto gc = std::make_shared<op::internal::GatherCompressed>(data, indices, axis, 0, scale);
    // auto model = std::make_shared<Model>(OutputVector{gc},
    //                                      ParameterVector{data, indices});

    // ov::Core core;
    // EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
    GTEST_SKIP() << "TODO: wire up GatherCompressed ctor + model, then enable EXPECT_THROW";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=CpuGatherCompressed.ZeroDimScaleShapeIsRejectedNotSIGFPE. Pre-fix expected failure: process aborts with SIGFPE / 'integer divide by zero' (floating point exception) inside Gather::initSupportedPrimitiveDescriptors at gather.cpp:206 (and :218 for the 5-input/zp variant). Post-fix expected: core.compile_model throws ov::Exception (CPU_NODE_ASSERT 'GATHER_SCALE input has zero elements'), satisfying EXPECT_THROW. (Skeleton: finalize the GatherCompressed ctor and includes from intel_cpu/tests/unit before running.)

## Suggested fix
Before line 205, assert (or throw) that the scale and zp element counts are non-zero:

```cpp
const size_t scale_count = getInputShapeAtPort(GATHER_SCALE).getElementsCount();
CPU_NODE_ASSERT(scale_count != 0, "GATHER_SCALE input has zero elements (zero-dim shape)");
scale_group_size = getInputShapeAtPort(GATHER_DATA).getElementsCount() / scale_count;
```

And similarly before line 217:

```cpp
const size_t zp_count = getInputShapeAtPort(GATHER_ZP).getElementsCount();
CPU_NODE_ASSERT(zp_count != 0, "GATHER_ZP input has zero elements (zero-dim shape)");
zp_group_size = getInputShapeAtPort(GATHER_DATA).getElementsCount() / zp_count;
```

Alternatively, call `getInputShapeAtPort(GATHER_SCALE).hasZeroDims()` (cpu_shape.h:157) and throw an error before entering the division.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #60.
