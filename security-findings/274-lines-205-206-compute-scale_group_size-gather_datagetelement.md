# Security finding #274: Lines 205-206 compute `scale_group_size = GATHER_DATA.getElementsCo…

**Summary:** Lines 205-206 compute `scale_group_size = GATHER_DATA.getElementsCo…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Divide-by-zero in `execCompressed8Bit` at lines 832-833 (`p / scale_group_size`, `p / zp_group_size`) and in `execCompressed4Bit` at lines 748-749, causing undefined behavior / SIGFPE crash. An attacker who can supply a crafted ONNX/IR model with a compressed-weight Gather node can trivially crash the inference engine (DoS). Depending on the platform and compiler, UB from size_t division-by-zero may also corrupt state.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-supplied GATHER_SCALE (and GATHER_ZP) tensor shapes — element counts are read directly from the model's static/inferred shapes without validating that scale count does not exceed data count

## Description / Root cause
Lines 205-206 compute `scale_group_size = GATHER_DATA.getElementsCount() / GATHER_SCALE.getElementsCount()` with plain integer division and no guard. If a hostile model provides a scale tensor whose element count is greater than the data tensor's element count, integer truncation yields `scale_group_size = 0`. The same applies to `zp_group_size` at lines 217-218. Neither `initSupportedPrimitiveDescriptors` nor `prepareParams` (lines 381-459) ever validates that the computed group sizes are ≥ 1 before they are used as divisors.

**Validator analysis:** The defect is real: at gather.cpp:205-206 scale_group_size = GATHER_DATA.getElementsCount() / GATHER_SCALE.getElementsCount() (and zp at 217-218) is plain size_t division with no check that the result is >=1 and no check that scale count <= data count. When scale count > data count the quotient truncates to 0. That 0 is later used as a divisor in the reference path (line 832 `p / scale_group_size`, 833 `p / zp_group_size`; 4-bit path 748-749) and even in the 'optimized' path (818/819) where additionally `p += scale_group_size` of 0 would loop forever — both reachable since neither prepareParams (381-459) nor the ctor (85-146) validates scale/data element-count relationship. CWE-369 Divide-By-Zero is the correct classification and the impact (SIGFPE/DoS on x86, UB generally) is accurate. Severity is moderate: it requires a hostile model whose decompression subgraph survives upstream transforms to materialize a GatherCompressed node with scale_count > data_count, which is atypical for legitimate quantization (scale normally <= data) but is not prevented anywhere on the path — that is the genuine gap. The proposed fix is correct and sufficient: guard scale_group_size==0 / zp_group_size==0 (or assert getElementsCount(scale) <= getElementsCount(data) and divisibility) at lines 206/218 with CPU_NODE_THROW, converting the crash into a clean rejection at primitive-descriptor init before any execution. Recommend also adding the divisibility check so partial groups (data%scale != 0) are rejected too.

## Exploit / Proof of Concept
Craft a model with a Gather node where `compressed=true`, the data tensor has shape [4] (4 elements) and the scale tensor has shape [8] (8 elements). `getElementsCount()` returns 4 and 8 respectively; integer division gives `scale_group_size = 0`. When inference is run, `execCompressed8Bit` is dispatched; the optimised path is skipped (axis != 0 or cond3 false), so execution falls to the reference path at line 832 where `p / scale_group_size` performs division-by-zero.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-369 in gather.cpp:205-206 (scale_group_size) and 217-218 (zp_group_size).
// Pre-fix: a GatherCompressed node whose SCALE tensor has MORE elements than the DATA tensor makes
//   scale_group_size = dataElems / scaleElems truncate to 0, then divided at gather.cpp:832-833 -> SIGFPE/UB.
// Post-fix: initSupportedPrimitiveDescriptors() must reject the node (CPU_NODE_THROW) before execution.
//
// Harness: ov_cpu_unit_tests (OpenVINO intel_cpu component tests).
// TODO: confirm exact test dir/helpers by reading openvino/src/plugins/intel_cpu/tests/unit/ and the
//       existing GatherCompressed/node single-layer test that builds an ov::op::internal::GatherCompressed.

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
// TODO: include the internal GatherCompressed op header used by the CPU plugin
// #include "ov_ops/gather_compressed.hpp"

// TODO: include the CPU-node test fixture that lets you build a single node and call
//       initSupportedPrimitiveDescriptors()/prepareParams()/execute() (see intel_cpu/tests/unit).

TEST(GatherCompressedCpuNode, RejectsScaleWithMoreElementsThanData_NoDivByZero) {
    // data: shape [4] -> 4 elements ; scale: shape [8] -> 8 elements (hostile, scale_group_size would be 0)
    // TODO: build Parameter(data u8 [4]), indices i32 [1], axis const 0, scale f32 [8] (+ optional zp f32 [8]),
    //       wrap into ov::op::internal::GatherCompressed, construct the CPU Gather node with a GraphContext,
    //       then assert the node refuses the configuration once the fix is applied.
    //
    // EXPECT_THROW(node.initSupportedPrimitiveDescriptors(), ov::Exception);   // passes only after fix
    //
    // Without the fix, execution of execCompressed8Bit() at gather.cpp:832 divides by scale_group_size==0,
    // which ov_cpu_unit_tests under ASan/UBSan reports as SIGFPE / 'division by zero'.
    GTEST_SKIP() << "TODO: needs GatherCompressed node builder + CPU node fixture wiring (see intel_cpu/tests/unit).";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ./ov_cpu_unit_tests --gtest_filter='GatherCompressedCpuNode.RejectsScaleWithMoreElementsThanData_NoDivByZero'. Pre-fix expectation (with -DENABLE_SANITIZER=ON / UBSan): SIGFPE or UBSan 'division by zero' at gather.cpp:832-833 (or 818). Post-fix: node throws ov::Exception at initSupportedPrimitiveDescriptors (gather.cpp:206/218) and the test passes.

## Suggested fix
After computing `scale_group_size` at line 206, add: `if (scale_group_size == 0) CPU_NODE_THROW("scale tensor has more elements than data tensor; scale_group_size would be zero");`. Apply the same guard after `zp_group_size` at line 218. Alternatively, enforce at this point that `GATHER_SCALE.getElementsCount() <= GATHER_DATA.getElementsCount()` and that `GATHER_DATA.getElementsCount() % GATHER_SCALE.getElementsCount() == 0`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #274.
