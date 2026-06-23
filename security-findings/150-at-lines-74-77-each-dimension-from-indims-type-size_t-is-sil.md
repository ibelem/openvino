# Security finding #150: At lines 74-77, each dimension from `inDims` (type `size_t`) is sil…

**Summary:** At lines 74-77, each dimension from `inDims` (type `size_t`) is sil…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound / CWE-787: Out-of-bounds Write
**Severity / Impact:** Signed 32-bit integer overflow is undefined behavior in C++; on all mainstream x86/x64 targets it produces a negative or unexpectedly small index. `dst_data[dstIndex]` at line 96 then writes a float to an address potentially gigabytes before or after the allocated output buffer — a heap memory corruption that can be leveraged for arbitrary code execution or process crash when loading a crafted model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:86` — `ReorgYolo::execute()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX/OpenVINO model file → tensor dimensions (B, IC, IH, IW) loaded as size_t StaticDims → narrowed to signed 32-bit int at lines 74-77 → used in 32-bit index arithmetic at line 86

## Description / Root cause
At lines 74-77, each dimension from `inDims` (type `size_t`) is silently narrowed to `int` with no range check. At line 86, `dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw` is computed entirely in signed 32-bit `int` arithmetic. No widening to `int64_t` or `size_t` occurs before or during the multiplications. There is no `OPENVINO_ASSERT` or bounds-check between this arithmetic and the write `dst_data[dstIndex]` at line 96.

**Validator analysis:** The flaw is real: at reorg_yolo.cpp:74-77 the four dimensions (IW/IH/IC/B) are stored as `int`, and at line 86 `int dstIndex = b * IC * IH * IW + ...` is evaluated in signed 32-bit arithmetic with no widening to int64_t/size_t and no OPENVINO_ASSERT before `dst_data[dstIndex]` at line 96. Signed overflow is UB and on x86 wraps to a negative index, yielding an OOB write — so CWE-190 leading to CWE-787 is correctly categorised. Reachability caveats: (1) The execute() path runs at INFERENCE time, not merely at model 'load' as the finding states — a minor wording error. (2) The PoC needs the output (and input) tensor of ~12 GB to actually allocate (so the OOB write lands in mapped/adjacent heap); on machines where that allocation fails the process simply throws/aborts rather than corrupting memory, so practical severity is lower than 'arbitrary code execution on load' — but the overflow itself is genuine and triggerable wherever such allocations succeed, and individual int truncation can also occur for dims >INT_MAX. (3) Reachable from the OpenVINO repo's trust boundary (IR model -> intel_cpu plugin), but NOT from the ORT OpenVINO EP, because no ONNX op lowers to ReorgYolo. The proposed fix (use size_t/int64_t for all dims and index temporaries, plus an OPENVINO_ASSERT that B*IC*IH*IW fits int64_t) is correct and sufficient; I would additionally compute srcIndex in the same widened type (the finding already mentions ih_off/iw_off/ic_off) and ensure the assert runs before any allocation/loop.

## Exploit / Proof of Concept
Craft a model with B=3, IC=32768, IH=32768, IW=1 (total 3×2^30 = ~12 GB elements; 64-bit allocation succeeds with size_t). At the loop iteration b=2, ic=0, ih=0, iw=0: `dstIndex = 2 * 32768 * 32768 * 1` = 2^31 = 2147483648, which overflows `int` (INT_MAX=2147483647) and wraps to -2147483648. `dst_data[-2147483648]` writes 8 GB before the output buffer start — a controlled heap underwrite reachable purely by loading the model.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190/CWE-787 in
//   openvino/src/plugins/intel_cpu/src/nodes/reorg_yolo.cpp:74-86,96
// where dim vars (IW/IH/IC/B) are `int` and `dstIndex = b*IC*IH*IW + ...`
// is computed in signed 32-bit arithmetic, so a product >= 2^31 wraps
// negative and `dst_data[dstIndex]` writes out of bounds.
//
// This assertion encodes the fix: after widening the index math to int64_t/
// size_t and adding an OPENVINO_ASSERT(B*IC*IH*IW <= INT64_MAX, ...), a
// ReorgYolo op whose total element count exceeds INT_MAX must be rejected
// (or processed without OOB write) instead of silently overflowing.
// Pre-fix: ASan reports a heap-buffer-overflow WRITE in ReorgYolo::execute.
// Post-fix: the op throws ov::Exception (or completes in-bounds).
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit.
//
// NOTE: a fully self-contained, *compilable* test that actually triggers the
// overflow is impractical — it requires ~2^31 output elements (~8 GB+ of
// allocation) for the multiply to cross INT_MAX. This is therefore emitted as
// a SKELETON; the real fix should add a cheap pre-loop range check that this
// test can exercise without giant allocations.

#include <gtest/gtest.h>
// TODO: include the intel_cpu node test scaffolding actually used by the
// existing tests under openvino/src/plugins/intel_cpu/tests/unit/ — read that
// directory to learn the exact fixture headers (e.g. the node/graph test
// helpers) and the correct namespace for ReorgYolo / Graph construction.

TEST(ReorgYoloOverflow, RejectsIndexExceedingInt32) {
    // TODO: build a single-node ReorgYolo graph with f32 ncsp input whose
    //       static dims make B*IC*IH*IW exceed INT_MAX, e.g.
    //       inDims = {3, 32768, 32768, 1}, stride = 1.
    //       Use the intel_cpu unit-test graph/node builder (look up the exact
    //       API in tests/unit/ — do NOT guess symbol names).
    //
    // TODO: after the fix adds OPENVINO_ASSERT on the total element count,
    //       expect construction/prepareParams/execute to throw:
    //
    //   EXPECT_THROW(node->execute(stream), ov::Exception);
    //
    // Pre-fix this path silently computes a negative dstIndex and writes OOB,
    // which ASan flags as a heap-buffer-overflow WRITE in ReorgYolo::execute
    // (reorg_yolo.cpp:96).
    GTEST_SKIP() << "Skeleton: fill in intel_cpu node builder + post-fix range "
                    "check assertion; see reorg_yolo.cpp:74-96";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter='ReorgYoloOverflow.*'. Expected pre-fix (once the skeleton is completed to actually run execute() with B*IC*IH*IW>INT_MAX and a successful allocation): AddressSanitizer 'heap-buffer-overflow WRITE of size 4' in ov::intel_cpu::node::ReorgYolo::execute at reorg_yolo.cpp:96. Expected post-fix: the op throws ov::Exception from the added OPENVINO_ASSERT and the test passes.

## Suggested fix
Change all four dimension variables to `size_t` (or `int64_t`) at lines 74-77: e.g. `size_t IW = (inDims.size() > 3) ? inDims[3] : 1;`. Change `dstIndex` and `srcIndex` (and `ih_off`, `iw_off`, `ic_off`) to `size_t` / `int64_t` throughout lines 79-94. Before the loops, add a range check: `OPENVINO_ASSERT(B * IC * IH * IW <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), ...)` and use `size_t` for all subscript arithmetic so no signed overflow is possible.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #150.
