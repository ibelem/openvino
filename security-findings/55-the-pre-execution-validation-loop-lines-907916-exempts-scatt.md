# Security finding #55: The pre-execution validation loop (lines 907–916) exempts ScatterEl…

**Summary:** The pre-execution validation loop (lines 907–916) exempts ScatterEl…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Heap out-of-bounds write before the data buffer. An attacker controlling a model's ScatterElementsUpdate indices tensor can corrupt arbitrary heap memory. This can lead to remote code execution or reliable process crash (denial-of-service) depending on heap layout. Any application loading untrusted ONNX models via OpenVINO CPU EP is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:605` — `ScatterUpdate::scatterElementsUpdate<DataType, KernelType>()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX model-supplied indices tensor ingested by OpenVINO CPU execution provider

## Description / Root cause
The pre-execution validation loop (lines 907–916) exempts ScatterElementsUpdate from the lower-bound check (line 914: `idxValue >= 0 || scatterUpdateMode == ScatterElementsUpdate`), so an index like `-data_dim_size - 1` passes that check. After normalization (`idxValue += data_dim_size`, line 603), the value becomes `-1` (still negative). The only remaining guard is `assert(idxValue < data_dim_size && idxValue >= 0)` at line 605, which is a no-op in release builds (NDEBUG). The pointer dereference at line 606 (`dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]`) then uses the negative `idxValue`, writing before the heap buffer. The same pattern is replicated at lines 627/628, 650/651, 669/670 in the generic template, and at lines 734/735, 758/759, 793/794, 814/815 in the ReduceMean specialisation.

**Validator analysis:** vulnType CWE-787 (Out-of-bounds Write) is accurate: a negative idxValue produces a negative byte offset from dataPtr, writing before the heap allocation. Impact (heap corruption → DoS/possible RCE) is plausible for any app loading untrusted ONNX via the OpenVINO CPU EP, though the write is at a fixed negative offset (idxValue can only reach -1 for the minimal -dim-1 case; larger magnitudes like -2*dim-1 give -dim-1 etc., so attacker control over the negative distance is bounded by the chosen index magnitude — still a definite OOB write). Root cause is the single, non-idempotent normalization combined with the validation exemption: the pre-exec check rejects idxValue >= srcDimAxis but, for SEU, accepts ANY negative value, while the kernel only clamps the [-dim,-1] range correctly. The proposed fix is correct and sufficient in principle: (1) tightening line 913-915 to require idxValue >= -srcDimAxis closes the entry path, and (2) replacing each kernel assert with a runtime range check defends in depth. I recommend BOTH: harden the pre-exec validation (preferred single chokepoint — change condition to `idxValue >= -static_cast<int64_t>(srcDimAxis) && idxValue < static_cast<int64_t>(srcDimAxis)` for the SEU branch) AND keep a runtime guard at each kernel site instead of assert (since the kernel can be reached for the use_init_val/reduction paths). A 'continue'/skip silently drops the element, which diverges from ONNX semantics; throwing (CPU_NODE_THROW) is the safer choice to fail closed.

## Exploit / Proof of Concept
Supply a ScatterElementsUpdate node where the data axis dimension is, say, N=10. Set an indices tensor containing the value `-N - 1` (e.g., -11). The pre-execution check at line 913 compares -11 < 10 (true) and skips the >= 0 check for ScatterElementsUpdate, so it passes. In the kernel, `if (idxValue < 0) idxValue += data_dim_size` yields -11 + 10 = -1. `assert(-1 >= 0)` is stripped in release. `dataPtr[offsets[0] + (-1) * dataBlock_axisplus1]` computes a negative offset from the base pointer (wrap via signed arithmetic promoted to pointer arithmetic), writing `dataBlock_axisplus1 * sizeof(DataType)` bytes before the allocation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:913-915 (validation exemption)
//   and the kernel deref at line 606 (assert is a no-op under NDEBUG).
//
// Encodes: a ScatterElementsUpdate model whose indices tensor contains a value
// < -data_dim_size (e.g. -(N+1) with N=axis-dim) must be REJECTED at execution
// (CPU_NODE_ASSERT -> ov::Exception), not silently dereferenced. Pre-fix this
// passes the validation loop, idxValue normalizes to -1, and ASan reports a
// heap-buffer-overflow write before the data buffer at scatter_update.cpp:606.
//
// Harness: ov_cpu_unit_tests (component target for openvino/src/plugins/intel_cpu).
// SKELETON: exact graph-build / infer-request helper symbols must be copied from
// the nearest existing tests under intel_cpu/tests/unit/ (e.g. the subgraph/
// single-layer test fixtures) — they are not guessed here.

#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test helpers actually used by the existing
//       tests in src/plugins/intel_cpu/tests/unit/ (graph builder + infer harness).
//       e.g. #include "nodes/..." or the test graph utility headers — READ that
//       dir to find the real fixture base class and helper names.

#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(ScatterElementsUpdate_CPU, RejectsIndexBelowNegativeDimSize) {
    // data: shape [10], axis = 0  -> data_dim_size = 10
    constexpr int64_t N = 10;
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{static_cast<size_t>(N)});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float>   dbuf(N, 0.0f);
    std::vector<int32_t> ibuf{ static_cast<int32_t>(-N - 1) }; // -11: passes buggy check, normalizes to -1
    std::vector<float>   ubuf{ 1.0f };

    req.set_input_tensor(0, Tensor(element::f32, Shape{static_cast<size_t>(N)}, dbuf.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{1}, ibuf.data()));
    req.set_input_tensor(2, Tensor(element::f32, Shape{1}, ubuf.data()));

    // Post-fix: out-of-range index is rejected before the kernel deref.
    // Pre-fix (release/NDEBUG): no throw and ASan reports heap-buffer-overflow WRITE
    //   at scatter_update.cpp:606.
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./bin/ov_cpu_unit_tests --gtest_filter='ScatterElementsUpdate_CPU.RejectsIndexBelowNegativeDimSize'. Expected pre-fix under ASan: 'heap-buffer-overflow ... WRITE of size N' originating in ScatterUpdate::scatterElementsUpdate at scatter_update.cpp:606 (and the same for ReduceMean specialization line 735). Expected post-fix: the index is rejected (ov::Exception) and ASSERT_ANY_THROW passes with no ASan report. NOTE: confirm the real fixture/helper symbols by reading src/plugins/intel_cpu/tests/unit/ before relying on this skeleton.

## Suggested fix
Replace the `assert` at each kernel site with a runtime range check and skip/throw on violation, e.g.: `if (idxValue < 0 || idxValue >= data_dim_size) { /* skip or throw */ continue; }`. Additionally, fix the pre-execution validation at line 913–915 to enforce the lower bound even for ScatterElementsUpdate: change the condition to `idxValue < static_cast<int64_t>(srcDimAxis) && idxValue >= -static_cast<int64_t>(srcDimAxis)` so that indices outside `[-data_dim_size, data_dim_size-1]` are rejected before reaching the kernel.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #55.
