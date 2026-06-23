# Security finding #54: Line 883 `pdst[pindices[i]] = pupdate[i]` uses a model-supplied `in…

**Summary:** Line 883 `pdst[pindices[i]] = pupdate[i]` uses a model-supplied `in…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Heap corruption / out-of-bounds write on the CPU EP heap. An attacker who controls the model (or its indices tensor) can overwrite arbitrary heap memory at a controlled 4-byte-aligned offset relative to `pdst`. Depending on heap layout this can achieve code execution, crash/DoS, or silent data corruption. Any process that loads an untrusted ONNX/IR model and runs it through the OpenVINO CPU EP is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883` — `ScatterUpdate::execute()`
**Validated for repos:** openvino
**Trust boundary:** ONNX/OpenVINO model-supplied indices tensor values ingested by the CPU EP at inference time

## Description / Root cause
Line 883 `pdst[pindices[i]] = pupdate[i]` uses a model-supplied `int32_t` index directly as an array subscript with no bounds check. The early-return 1-D fast path (lines 867–887) returns before the index validation loop at lines 907–917. The only guard is `srcDataDim[0] <= 64` (line 869), which caps the *buffer size* to ≤ 64 `int32_t` elements, but imposes no constraint on the *value* of `pindices[i]`. A negative `pindices[i]` is implicitly converted to a large `size_t`/`ptrdiff_t` offset writing before `pdst`; a value ≥ srcLength writes past the buffer.

**Validator analysis:** The defect is genuine: in the 1-D fast path of ScatterUpdate::execute (lines 867-885), a model-supplied int32_t index `pindices[i]` is used directly as a subscript into the int32_t buffer `pdst` at line 883 with no bounds/sign check, and the function `return`s at line 885 before reaching the general-path index validation at lines 913-915. The guard at line 869 (`srcDataDim[0] <= 64`) only caps the destination buffer to <=64 int32 elements; it does not constrain the *value* of the index. A negative index writes before `pdst`; an index >= srcLength writes past it — a true CWE-787 OOB write of attacker-controlled 4-byte data at a controlled offset. The vuln_type (CWE-787) and impact (heap corruption / potential RCE/DoS) are accurate. HOWEVER the stated trust boundary ('ONNX/OpenVINO model-supplied indices ingested by the CPU EP') is only half right: the fast path is gated on scatterUpdateMode==ScatterUpdate, which is set exclusively for ov::op::v3::ScatterUpdate (constructor lines 117-118). The ONNX frontend never emits a v3 ScatterUpdate node — Scatter/ScatterElements -> v12::ScatterElementsUpdate (ScatterElementsUpdate mode) and ScatterND -> ScatterNDUpdate (scatter_elements.cpp:45) — so an attacker confined to ONNX (the openvinoEp boundary) cannot reach this code, hence openvinoEp is rejected. A native OpenVINO IR model can contain v3 ScatterUpdate directly, so the OpenVINO core boundary is affected. The proposed fix is correct and sufficient: adding `CPU_NODE_ASSERT(idx >= 0 && static_cast<size_t>(idx) < srcLength, ...)` inside the loop before the write mirrors the existing 913-915 validation and closes both the negative and overflow cases. One refinement: srcLength is `srcMemPtr->getStaticDims()[0]`, which equals the dst length here, so the bound is correct; the assert should use srcLength (the dst element count) exactly as written.

## Exploit / Proof of Concept
Supply a ScatterUpdate node with: data shape [4] (int32, 4 elements ≤ 64 ✓), indices shape [1] (int32, scalar/1-D ✓), update shape [1] (int32 ✓). Set the runtime indices tensor value to e.g. -1 or 1000. The fast-path condition at line 868 is satisfied; execution enters the loop at line 882; `pindices[0]` = -1 or 1000 is used directly in `pdst[-1]` or `pdst[1000]`, writing `pupdate[0]` to a heap location 4 bytes before or 3984 bytes past the 16-byte `pdst` buffer.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for the OOB write at
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883
//   pdst[pindices[i]] = pupdate[i];   // model-supplied int32 index, no bounds check
// The 1-D fast path (lines 867-885) returns before the general-path index
// validation at lines 913-915, and the only guard (srcDataDim[0] <= 64, line 869)
// bounds the buffer size, not the index VALUE.
//
// This test builds a v3::ScatterUpdate model that hits the fast path
// (data i32 shape [4] <= 64, indices i32 shape [1], update i32 shape [1]) and
// feeds an out-of-range index value (1000). Pre-fix: ASan reports a heap OOB
// write inside ScatterUpdate::execute. Post-fix: the CPU_NODE_ASSERT throws an
// ov::Exception, so infer_request.infer() throws and ASSERT_ANY_THROW passes.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). Run on CPU plugin.
// TODO: confirm the exact include paths/helpers against a sibling test under
//       intel_cpu/tests/unit/ (e.g. an existing model-construction test) — the
//       symbol names below are best-effort and may need adjustment.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

TEST(ScatterUpdateCpuOOB, FastPathRejectsOutOfRangeIndex) {
    // data: i32 [4]  (<=64 -> enters the 1-D fast path)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // indices: i32 [1]
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // updates: i32 [1]
    auto updates = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // axis = 0
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> data_vals{0, 0, 0, 0};
    std::vector<int32_t> idx_vals{1000};   // out of range for a 4-element dst
    std::vector<int32_t> upd_vals{0x41414141};

    Tensor t_data(element::i32, Shape{4}, data_vals.data());
    Tensor t_idx(element::i32, Shape{1}, idx_vals.data());
    Tensor t_upd(element::i32, Shape{1}, upd_vals.data());

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_idx);
    req.set_input_tensor(2, t_upd);

    // Pre-fix: ASan aborts on heap-buffer-overflow at scatter_update.cpp:883.
    // Post-fix: CPU_NODE_ASSERT throws ov::Exception, surfaced by infer().
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=ScatterUpdateCpuOOB.FastPathRejectsOutOfRangeIndex. Pre-fix with -DENABLE_SANITIZER=ON expect AddressSanitizer 'heap-buffer-overflow WRITE of size 4' inside ScatterUpdate::execute (scatter_update.cpp:883); post-fix expect a clean pass because the in-loop CPU_NODE_ASSERT throws ov::Exception caught by ASSERT_ANY_THROW. NOTE: verify CPU plugin device name and Tensor/InferRequest helper usage against a sibling test in intel_cpu/tests/unit before running.

## Suggested fix
Insert a bounds check inside the fast-path loop before the write. For example:
```cpp
for (size_t i = 0; i < updateCnt; i++) {
    int32_t idx = pindices[i];
    CPU_NODE_ASSERT(idx >= 0 && static_cast<size_t>(idx) < srcLength,
        "have indices value that points to non-existing output tensor element");
    pdst[idx] = pupdate[i];
}
```
This mirrors the validation already present for the general path at lines 913–915 and keeps the fast path safe.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #54.
