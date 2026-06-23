# Security finding #146: The 1D fast-path loop at lines 882–884 uses `pindices[i]` (a model-…

**Summary:** The 1D fast-path loop at lines 882–884 uses `pindices[i]` (a model-…

**CWE IDs:** CWE-129: Improper Validation of Array Index / CWE-787: Out-of-bounds Write
**Severity / Impact:** Out-of-bounds write to adjacent heap memory with fully attacker-controlled 32-bit value and 32-bit data. Exploitable for heap metadata corruption leading to arbitrary code execution or reliable crash/DoS. Affects any application loading a crafted ONNX/OpenVINO model that triggers the ScatterUpdate 1D fast-path (srcDataDim[0] ≤ 64, indices rank ≤ 1, int32 data/indices).
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:882` — `ScatterUpdate::execute()`
**Validated for repos:** openvino
**Trust boundary:** INDICES_ID input tensor provided by untrusted model (crosses the model-load trust boundary)

## Description / Root cause
The 1D fast-path loop at lines 882–884 uses `pindices[i]` (a model-supplied int32_t) as a raw array subscript into `pdst` (`int32_t*`) with no bounds check. The slow/general path (lines 913–915) guards equivalent index consumption with a CPU_NODE_ASSERT that rejects values outside `[0, srcDimAxis)`, but this fast-path has no such guard. `pdst` is allocated for exactly `srcDataDim[0]` elements (guaranteed ≤ 64 by line 869). A negative int32_t index is implicitly converted to a large negative ptrdiff_t offset (writing before the buffer), while any index ≥ srcDataDim[0] writes past the end.

**Validator analysis:** The defect is real and accurately categorised: lines 882-884 use the untrusted int32 pindices[i] as a raw subscript into pdst (allocated for srcDataDim[0]<=64 int32 elements at 876-878) with zero validation, while the equivalent general-path consumption at 913-915 is guarded by CPU_NODE_ASSERT against [0, srcDimAxis). A negative or >=srcDataDim[0] index yields an OOB heap write of an attacker-controlled 32-bit value (CWE-129/CWE-787) — impact (heap corruption / crash / potential RCE) is plausible. However the trust-boundary reachability differs per repo: the fast path is gated on scatterUpdateMode==ScatterUpdate, i.e. the OpenVINO opset-v3 ScatterUpdate operator. That operator is producible directly in an OpenVINO IR/model (openvino boundary = validated) but ONNX has no operator that lowers to v3 ScatterUpdate; ScatterElements maps to ScatterElementsUpdate (a different mode that does not take this branch) and ScatterND maps to ScatterNDUpdate. Without a demonstrated ConvertScatterElementsToScatter rewrite whose preconditions are met, reachability from the ORT OpenVINO-EP boundary is unproven, so openvinoEp is rejected (not na — the node exists in the shared plugin, just not shown reachable from ONNX). The proposed fix is correct and sufficient: mirroring the slow-path assertion (index>=0 && (size_t)index<srcDataDim[0]) before the write loop closes the hole; one refinement — bound the copy against srcLength and also validate when updateDims is scalar (updateCnt==1) which the loop already covers. The CPU_NODE_ASSERT form matches the existing error-conversion boundary and will turn the OOB write into a clean ov::Exception.

## Exploit / Proof of Concept
Craft an ONNX model with a ScatterUpdate node where: DATA has shape [16] (int32, 16 ≤ 64 satisfies fast-path), INDICES is a 1-element int32 tensor with value -1 (or 17, 100, etc.), UPDATES is a 1-element int32 tensor with an attacker-chosen value. On execute(), line 880 casts indicesPtr to `int32_t*`, and line 883 evaluates `pdst[-1] = pupdate[0]` — a write 4 bytes before the `dstPtr` allocation — corrupting heap metadata or an adjacent object. Alternatively, index 0x7fffffff gives a 2 GB forward write. No validator between the model-load boundary and line 882 rejects the index value.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:882-884 (ScatterUpdate 1D fast-path OOB write).
// Pre-fix: an out-of-range INDICES value drives pdst[pindices[i]]=pupdate[i] past the
//          dst allocation -> ASan heap-buffer-overflow (or silent corruption).
// Post-fix: the mirrored CPU_NODE_ASSERT(pindices[i] in [0,srcDataDim[0])) rejects it,
//          surfacing as an ov::Exception at inference time.
//
// NOTE: ov_cpu_unit_tests is primarily a node-level harness; reaching the fast-path
// cleanly is easiest via a compiled CPU model that exercises opset3 ScatterUpdate with
// a 1D i32 DATA of length <= 64 and a malicious INDICES constant. Symbols below
// (ov::Core, opset3) are standard; the exact include/test-registration may need tuning
// against the surrounding tree, hence this is a SKELETON.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset3.hpp"

using namespace ov;

TEST(ScatterUpdateCpu1DFastPath, RejectsOutOfRangeIndex) {
    // DATA: 1D i32, length 16 (<=64) -> satisfies fast-path predicate at scatter_update.cpp:868-869
    auto data = std::make_shared<opset3::Parameter>(element::i32, Shape{16});
    // INDICES: single i32 index = 17 (>= srcDataDim[0]==16) -> OOB write pre-fix
    auto indices = opset3::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{17});
    auto axis    = opset3::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto updates = opset3::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{0x7fffffff});

    auto su = std::make_shared<opset3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(NodeVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{16});
    std::fill_n(in.data<int32_t>(), 16, 0);
    req.set_input_tensor(in);

    // TODO: if the fast-path requires a const-folded DATA path or shape-inference subgraph
    //       context to be selected (see comment "optimized for shape inference subgraph"),
    //       wrap DATA accordingly so exec1DCase is chosen.
    EXPECT_THROW(req.infer(), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests ; run: ./ov_cpu_unit_tests --gtest_filter=ScatterUpdateCpu1DFastPath.RejectsOutOfRangeIndex . Pre-fix expectation under -DENABLE_SANITIZER=ON: ASan 'heap-buffer-overflow WRITE of size 4' at scatter_update.cpp:883 (or no throw -> test fails). Post-fix: req.infer() throws ov::Exception ('index out of range') and the test passes.

## Suggested fix
Mirror the slow-path validation immediately before the write loop. Add a check such as:
```cpp
for (size_t i = 0; i < updateCnt; i++) {
    CPU_NODE_ASSERT(pindices[i] >= 0 && static_cast<size_t>(pindices[i]) < srcDataDim[0],
        "ScatterUpdate 1D fast-path: index out of range");
    pdst[pindices[i]] = pupdate[i];
}
```
This mirrors the assertion at lines 913–915 and keeps the fast-path behaviour consistent with the general path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #146.
