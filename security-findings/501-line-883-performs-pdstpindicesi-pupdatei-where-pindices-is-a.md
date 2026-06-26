# Security finding #501: Line 883 performs `pdst[pindices[i]] = pupdate[i]` where `pindices`…

**Summary:** Line 883 performs `pdst[pindices[i]] = pupdate[i]` where `pindices`…

**CWE IDs:** CWE-129: Improper Validation of Array Index / CWE-787: Out-of-bounds Write
**Severity / Impact:** An attacker who controls the INDICES tensor (e.g. via a crafted ONNX model) can write an arbitrary `int32_t` value to an arbitrary int32-aligned location relative to the destination tensor buffer. A negative index writes before the buffer; a large positive index writes after it. The destination buffer is at most 64×4 = 256 bytes (enforced by `srcDataDim[0] <= 64`), so the write can reach adjacent heap metadata or other tensors. This enables heap corruption, crash (DoS), or — with sufficient heap-layout control — arbitrary code execution in the inference process.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883` — `ScatterUpdate::execute()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled INDICES input tensor loaded from a malicious ONNX model file

## Description / Root cause
Line 883 performs `pdst[pindices[i]] = pupdate[i]` where `pindices` is a raw `int32_t*` cast over the untrusted INDICES tensor buffer. No bounds check ([0, srcLength)) is applied to `pindices[i]` anywhere on this code path. The function returns at line 885 before reaching the validation loop at lines 911-916, which would have checked `idxValue < srcDimAxis && idxValue >= 0`.

**Validator analysis:** The flaw is real: the 1-D short-vector fast path (scatter_update.cpp:868-886) dereferences pdst[pindices[i]] using an unvalidated int32 index taken directly from the untrusted INDICES buffer, and returns at line 885 before the bounds assertion at 913-915 that guards the general path. A negative or large index yields an OOB write of an int32 relative to the destination buffer (CWE-129/CWE-787 — vuln type and impact are accurate). However the path is gated on scatterUpdateMode==ScatterUpdate (OV opset3 ScatterUpdate op), which the ONNX frontend does not produce (ONNX ScatterElements/ScatterND map to ScatterElementsUpdate/ScatterNDUpdate), so it is NOT reachable from the ONNX-EP trust boundary — hence openvinoEp is rejected and the exploit's 'crafted ONNX ScatterUpdate node' premise is incorrect. It remains reachable for openvino via a crafted IR model that declares a v3 ScatterUpdate op, so openvino is validated. The proposed fix is correct and sufficient: add `idx>=0 && (size_t)idx < srcLength` (where srcLength==srcDataDim[0]) before the write, mirroring the 913-915 assertion; one refinement — also guard that updateCnt does not exceed indices length (pindices[i] is read up to updateCnt, which derives from updateDims, not indicesDim), to avoid an OOB *read* of pindices when update is longer than indices.

## Exploit / Proof of Concept
Craft an ONNX ScatterUpdate node with: DATA shape [4] (i32), INDICES shape [1] (i32) containing value -1 or 200, UPDATE shape [1] (i32) containing any value, no AXIS input (axisRelaxed = false). When the model is loaded and inferred by OpenVINO's CPU EP, `ScatterUpdate::execute` is called; the fast-path conditions at line 868 are all satisfied (ScatterUpdate mode, 1-D data of size 4 ≤ 64, 1-D indices, i32 precision). The loop at line 882 executes `pdst[-1]` or `pdst[200]`, writing the update value 4 bytes before the destination buffer or 784 bytes past its end, respectively, then returns — validation at line 911 is never reached.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:882-885 (missing bounds check on the
// 1-D i32 short-vector ScatterUpdate fast path). Pre-fix: pdst[pindices[i]]
// with an out-of-range index performs an OOB heap write (ASan heap-buffer-overflow).
// Post-fix: the node must reject the out-of-bounds index via CPU_NODE_ASSERT
// (ov::Exception) instead of writing past the destination buffer.
//
// Harness: ov_cpu_unit_tests (intel_cpu/tests/unit). SKELETON — exact graph/
// infer-request helper symbols must be confirmed against the existing unit tree
// before use.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"

using namespace ov;

TEST(ScatterUpdateCpu1DFastPath, OutOfRangeIndexIsRejected) {
    // TODO: confirm correct test fixture/base in intel_cpu/tests/unit and the
    //       canonical way to build+compile a single-op model on the CPU plugin.

    // Build: data[4] (i32), indices[1] (i32), axis const(0), update[1] (i32)
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto update  = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, update, axis);
    auto model = std::make_shared<Model>(OutputVector{su->output(0)},
                                         ParameterVector{data, indices, update});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::i32, Shape{4});
    std::fill_n(t_data.data<int32_t>(), 4, 0);
    Tensor t_idx(element::i32, Shape{1});
    t_idx.data<int32_t>()[0] = 200;   // out of [0,4) -> OOB write pre-fix
    Tensor t_upd(element::i32, Shape{1});
    t_upd.data<int32_t>()[0] = 0x41414141;

    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_idx);
    req.set_input_tensor(2, t_upd);

    // Pre-fix: ASan reports heap-buffer-overflow on pdst[200] at scatter_update.cpp:883.
    // Post-fix: index is range-checked and infer() throws ov::Exception.
    EXPECT_ANY_THROW(req.infer());

    // TODO: also add a negative-index case (t_idx=-1) which writes before the buffer.
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=ScatterUpdateCpu1DFastPath.OutOfRangeIndexIsRejected . Pre-fix with -DENABLE_SANITIZER=ON expect ASan 'heap-buffer-overflow WRITE of size 4' originating at intel_cpu/src/nodes/scatter_update.cpp:883 (pdst[pindices[i]]); post-fix expect the test to pass because infer() throws ov::Exception from the added CPU_NODE_ASSERT.

## Suggested fix
Add an explicit range check inside the `updateCnt` loop before the write at line 883. For example:
```cpp
for (size_t i = 0; i < updateCnt; i++) {
    int32_t idx = pindices[i];
    CPU_NODE_ASSERT(idx >= 0 && static_cast<size_t>(idx) < srcLength,
                    "scatter index out of bounds");
    pdst[idx] = pupdate[i];
}
```
This mirrors the bounds assertion already applied in the general path at lines 913-915, adapted for the known `srcLength` available on this path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #501.
