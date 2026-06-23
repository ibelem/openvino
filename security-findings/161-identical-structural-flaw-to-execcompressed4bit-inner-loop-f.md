# Security finding #161: Identical structural flaw to execCompressed4Bit: inner loop `for (s…

**Summary:** Identical structural flaw to execCompressed4Bit: inner loop `for (s…

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-787: Out-of-bounds Write
**Severity / Impact:** Same as the 4-bit variant: heap OOB read (information disclosure of adjacent heap contents) and heap OOB write (memory corruption). Triggered at inference time by any model where the 8-bit compressed Gather node has `afterAxisSize` not divisible by `scale_group_size`. Potential RCE via heap corruption.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:820` — `Gather::execCompressed8Bit()`
**Validated for repos:** openvino
**Trust boundary:** Model file / ONNX graph: shapes of the compressed data tensor and the scale tensor are attacker-controlled and determine scale_group_size and afterAxisSize

## Description / Root cause
Identical structural flaw to execCompressed4Bit: inner loop `for (size_t g = p; g < p + scale_group_size; g++)` at line 820 does not clamp `g` to `srcIdx + afterAxisSize`. When `afterAxisSize % scale_group_size != 0`, the last outer-loop iteration causes `srcData[g]` (line 822) to read beyond the valid per-row source data and `pdst[dst_idx]` (line 821) to write beyond the per-row destination buffer.

**Validator analysis:** Vuln type CWE-125/CWE-787 is accurate: when afterAxisSize % scale_group_size != 0, the last outer iteration reads srcData[g] past the per-row source (OOB read, info disclosure of adjacent heap) and writes pdst[dst_idx] past the per-(b,j) destination region whose stride is afterAxisSize (OOB write, heap corruption). Reachable because the grouped path (lines 815-826) executes for non-scalar scale (cond3 false) with cond2 true (no zp / scalar zp), and scale_group_size is derived solely from attacker-controlled data/scale element counts. The scalar path (811) and reference path (831) are already correctly bounded to srcIdx+afterAxisSize, confirming the asymmetry is the defect. Proposed fix (clamp to std::min(p + scale_group_size, srcIdx + afterAxisSize) at line 820) is correct and sufficient to eliminate the OOB; the additional divisibility precheck in initSupportedPrimitiveDescriptors/prepareParams is a sound defense-in-depth but the min-clamp alone prevents the overrun. Same fix should be applied to all three group-decompression sites (cond1||cond2 branches) for both 8-bit and 4-bit variants.

## Exploit / Proof of Concept
Load a model with an INT8-compressed Gather node with data shape [2, 7], scale shape [2, 3] (scale_group_size = 7*2 / (3*2) truncates to 2 due to integer division; or arrange exact shapes so scale_group_size=3 and afterAxisSize=7, 7%3=1). The outer loop iterates p=srcIdx, srcIdx+3 (if scale_group_size=3), and srcIdx+6. At p=srcIdx+6, the inner loop runs g=srcIdx+6, srcIdx+7, srcIdx+8 — reads srcData[srcIdx+7] and srcData[srcIdx+8] (beyond the 7-element row) and writes pdst[7] and pdst[8] past the destination row.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB in Gather::execCompressed8Bit grouped decompression path.
// Encodes the fix for openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:820 — the inner
// loop `for (size_t g = p; g < p + scale_group_size; g++)` must be clamped to
// srcIdx + afterAxisSize so that when afterAxisSize % scale_group_size != 0 the last
// outer iteration does not read srcData past the row nor write pdst past the per-row
// destination stride. Pre-fix: ASan heap-buffer-overflow (read+write) during execute().
// Post-fix: graph runs and produces correctly sized output without OOB.
//
// Harness: ov_cpu_unit_tests (gtest + ASan). Place near intel_cpu/tests/unit node tests.
// NOTE (skeleton): the internal op ov::op::internal::GatherCompressed is normally produced
// by OpenVINO's weight-decompression fusion, not constructed directly in most unit tests.
// The exact builder helpers / fixture base must be confirmed against the existing
// intel_cpu/tests/unit tree before this compiles.

#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// TODO: confirm the correct fixture/base class and compile target name in
//       src/plugins/intel_cpu/tests/unit/ (e.g. an existing GatherCompressed node test).
TEST(GatherCompressed8Bit_OOB, NonDivisibleScaleGroupSize_NoHeapOverflow) {
    // data: int8 [axisDim=2, afterAxisSize=7]; axis=0; indices select a row.
    // scale shape [2,3] -> scale_group_size = (2*7)/(2*3) = 2 (integer division),
    // afterAxisSize=7, 7 % 2 == 1 -> triggers the unclamped inner loop overrun pre-fix.
    const Shape data_shape{2, 7};
    const Shape scale_shape{2, 3};

    // TODO: replace constant payloads with the precise dtype/precision the CPU plugin
    //       routes through execCompressed8Bit (i8 data, f32 scale, no zero-point => 4 inputs,
    //       non-scalar scale so cond3 is false and the grouped cond2 path is taken).
    auto data  = op::v0::Constant::create(element::i8,  data_shape,  std::vector<int8_t>(2 * 7, 1));
    auto idx   = op::v0::Constant::create(element::i32, Shape{1},    std::vector<int32_t>{0});
    auto axis  = op::v0::Constant::create(element::i32, Shape{},     std::vector<int32_t>{0});
    auto scale = op::v0::Constant::create(element::f32, scale_shape, std::vector<float>(2 * 3, 1.0f));

    auto gc = std::make_shared<op::internal::GatherCompressed>(
        data, idx, axis, /*batch_dims=*/0, scale);

    auto model = std::make_shared<Model>(OutputVector{gc}, ParameterVector{});

    Core core;
    // Pre-fix: ASan reports heap-buffer-overflow inside Gather::execCompressed8Bit.
    // Post-fix: compilation + inference succeed with no OOB.
    EXPECT_NO_THROW({
        auto compiled = core.compile_model(model, "CPU");
        auto req = compiled.create_infer_request();
        req.infer();
    });
}
```
**Build / run:** Build target: ov_cpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter=GatherCompressed8Bit_OOB.NonDivisibleScaleGroupSize_NoHeapOverflow . Expected pre-fix: ASan 'heap-buffer-overflow READ/WRITE' raised from Gather::execCompressed8Bit (gather.cpp:821-822); post-fix: test passes with no sanitizer report. (Skeleton: verify GatherCompressed builder usage and that the CPU plugin selects the grouped decompression path before relying on this test.)

## Suggested fix
Change `g < p + scale_group_size` to `g < std::min(p + scale_group_size, srcIdx + afterAxisSize)` at line 820. Apply the same divisibility pre-check validation in `initSupportedPrimitiveDescriptors` / `prepareParams` as described for the 4-bit variant.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #161.
