# Security finding #160: The inner loop `for (size_t g = p; g < p + scale_group_size; g++)` …

**Summary:** The inner loop `for (size_t g = p; g < p + scale_group_size; g++)` …

**CWE IDs:** CWE-125: Out-of-bounds Read / CWE-787: Out-of-bounds Write
**Severity / Impact:** Heap out-of-bounds read leaks adjacent memory (weight bytes, other tensor data); heap out-of-bounds write corrupts adjacent allocations. Both are reachable whenever a specially crafted model produces `afterAxisSize % scale_group_size != 0` and triggers the `cond1 || cond2` fast path. An attacker who controls the model file can trigger this at inference time, potentially escalating to RCE via heap metadata corruption.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:735` — `Gather::execCompressed4Bit()`
**Validated for repos:** openvino
**Trust boundary:** Model file / ONNX graph: shapes of the compressed data tensor and the scale tensor are attacker-controlled and determine scale_group_size and afterAxisSize

## Description / Root cause
The inner loop `for (size_t g = p; g < p + scale_group_size; g++)` at line 735 unconditionally iterates a full scale_group_size window starting at `p`. The outer loop condition `p < srcIdx + afterAxisSize` (line 732) allows the last outer iteration to start at `p` where `p + scale_group_size > srcIdx + afterAxisSize` when `afterAxisSize % scale_group_size != 0`. Inside that last iteration, `g` walks past `srcIdx + afterAxisSize`, causing `srcData[g >> 1]` (line 736) to read beyond the valid per-row source slice and `pdst[dst_idx]` (line 737) to write beyond the per-row destination allocation. No divisibility check (`afterAxisSize % scale_group_size == 0`) exists in `initSupportedPrimitiveDescriptors` or `prepareParams`.

**Validator analysis:** vulnType CWE-125/CWE-787 is accurate: when afterAxisSize is not a multiple of scale_group_size, ceil(afterAxisSize/scale_group_size)*scale_group_size > afterAxisSize, so dst_idx overruns the per-row destination (corruption, and a genuine heap OOB write on the final dst slice) and g overruns the per-row source nibble range (OOB read on the final data slice). Reachability is real: cond1||cond2 with non-scalar scale (cond3 false) is precisely the standard grouped-quantization embedding decompression path with isAxisInputConst && axis==0; only the non-divisible group size is attacker-supplied, and GatherCompressed::validate_and_infer_types only runs base-gather shape inference on inputs 0-2, never validating the scale/zp shape. The RCE-via-heap-metadata escalation in the impact is speculative but the OOB read/write themselves are sound. The proposed fix is correct and sufficient: clamping the inner bound to std::min(p + scale_group_size, srcIdx + afterAxisSize) eliminates both the OOB read and the OOB write while leaving dst_idx capped at afterAxisSize. The same clamp should also be applied if any sibling kernel (e.g. execCompressed8Bit at gather.cpp:766+, which mirrors this structure) shares the pattern. The early divisibility validation in initSupportedPrimitiveDescriptors/prepareParams is good defense-in-depth (reject malformed models with a clear ov::Exception rather than silently truncating scale_group_size via integer division), but is not strictly required once the loop is clamped.

## Exploit / Proof of Concept
Load a model where the Gather-compressed node has a data tensor with shape [N, K] and a scale tensor with shape [N, K/G] where G (scale_group_size = K*N / (K/G * N) = G) does not divide `afterAxisSize` (= K for axis=0). E.g., data shape [2, 7], scale shape [2, 3] gives scale_group_size=2 and afterAxisSize=7; 7%2=1 != 0. The outer loop's last iteration starts at p=6 (srcIdx+6), the inner loop runs g=6 and g=7, reading srcData[3] and srcData[3] (fine) then writing pdst[6] and pdst[7] — pdst[7] is one element past the 7-element destination row.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read/write in Gather::execCompressed4Bit
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:735-737
// Pre-fix: with afterAxisSize % scale_group_size != 0 the inner loop
//   `for (g = p; g < p + scale_group_size; g++)` overruns the per-row source
//   slice (srcData[g>>1], line 736) and destination slice (pdst[dst_idx],
//   line 737); on the final slice this is a heap OOB read + write that ASan
//   reports. Post-fix (loop clamped to min(p+scale_group_size, srcIdx+afterAxisSize)
//   and/or divisibility validation) the model is rejected or runs in-bounds.
//
// SKELETON: building a GatherCompressed internal op + running CPU inference
// requires the intel_cpu graph test fixtures; exact symbols must be taken from
// the surrounding test tree before this will compile.
//
// Harness: ov_cpu_unit_tests (gtest + ASan), style of
//   targets/openvino/src/plugins/intel_cpu/tests/unit/

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_compressed.hpp"
// TODO: include the CPU-plugin graph/infer harness headers used by the
//       existing tests under intel_cpu/tests/unit/ (e.g. the helper that
//       compiles an ov::Model on the CPU device and runs a single infer).

using namespace ov;

TEST(GatherCompressed4Bit, NonDivisibleGroupSizeIsRejectedOrInBounds) {
    // data: u4 [axisDim=2, afterAxisSize=7]  (7 nibbles per row)
    // scale: f32 [2, 3] -> scale_group_size = (2*7)/(2*3) = 2, and 7 % 2 == 1
    // indices: [0] on axis 0 -> selects a full 7-element row, drives the
    //          (cond1||cond2) grouped path with non-scalar scale (cond3 false).
    // TODO: construct the parameters/constants with the exact builder helpers
    //       used by the CPU unit tests:
    //   auto data    = <u4 Parameter/Constant, shape {2,7}>;
    //   auto indices = op::v0::Constant({0});           // axis-0 row select
    //   auto axis    = op::v0::Constant(0);
    //   auto scale   = <f32 Constant, shape {2,3}>;     // non-divisible group
    //   auto gc = std::make_shared<op::internal::GatherCompressed>(
    //                 data, indices, axis, /*batch_dims=*/0, scale);
    //   auto model = std::make_shared<ov::Model>(gc, ParameterVector{...});
    //
    // Pre-fix expectation: compiling+infer on CPU triggers ASan
    //   heap-buffer-overflow at gather.cpp:736/737.
    // Post-fix expectation: either the malformed group size is rejected at
    //   compile time (ov::Exception) or inference completes in-bounds.
    //
    // TODO: replace with the harness call, e.g.:
    //   EXPECT_NO_THROW({ auto compiled = core.compile_model(model, "CPU");
    //                     auto req = compiled.create_infer_request();
    //                     req.infer(); });
    // (the real assertion is that ASan does NOT fire; if the chosen fix
    //  rejects malformed shapes instead, switch to EXPECT_THROW(..., ov::Exception)).
    GTEST_SKIP() << "TODO: wire up CPU compile_model/infer harness and u4 data init";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=GatherCompressed4Bit.NonDivisibleGroupSizeIsRejectedOrInBounds . Expected pre-fix: ASan 'heap-buffer-overflow ... READ/WRITE' inside ov::intel_cpu::node::Gather::execCompressed4Bit (gather.cpp:736/737). Expected post-fix: test passes (no ASan report), or the malformed model is rejected via ov::Exception if the divisibility check is added.

## Suggested fix
Clamp the inner loop upper bound: change `g < p + scale_group_size` to `g < std::min(p + scale_group_size, srcIdx + afterAxisSize)`. Additionally, add a runtime assert (or validation error) in `initSupportedPrimitiveDescriptors`/`prepareParams` that `afterAxisSize % scale_group_size == 0` when the compressed fast-path is selected, rejecting malformed models early.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #160.
