# Security finding #502: The CPU_NODE_ASSERT at lines 913-914 checks `idxValue < srcDimAxis …

**Summary:** The CPU_NODE_ASSERT at lines 913-914 checks `idxValue < srcDimAxis …

**CWE IDs:** CWE-129: Improper Validation of Array Index
**Severity / Impact:** An attacker supplying a deeply negative index (e.g. -1000 on an axis of size 10) bypasses the only compile-time guard, causing the correction `idxValue += data_dim_size` inside scatterElementsUpdate to leave idxValue still negative. In release builds the sole remaining guard (`assert(idxValue >= 0)` at lines 605, 627, 650, 669, 734, 758, 793, 814) is a no-op (NDEBUG). The negative int64_t is then multiplied with `dataBlock_axisplus1` (size_t), and C++ usual-arithmetic-conversion promotes the result to a huge uint64_t. The computed offset `offsets[0] + huge_value` is used to write into `dataPtr[]` — an out-of-bounds write (CWE-787) that can corrupt heap metadata, neighboring objects, or enable arbitrary write primitives under attacker-controlled model execution.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:913` — `ScatterUpdate::execute()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled INDICES tensor loaded from a model file and fed to the ScatterElementsUpdate CPU node

## Description / Root cause
The CPU_NODE_ASSERT at lines 913-914 checks `idxValue < srcDimAxis && (idxValue >= 0 || scatterUpdateMode == ScatterElementsUpdate)`. For ScatterElementsUpdate mode the OR makes the lower-bound clause `(false || true)` unconditionally true, so any negative index — no matter how large in magnitude — passes without error. There is no check that `idxValue >= -srcDimAxis`.

**Validator analysis:** Confirmed real and reachable. At scatter_update.cpp:913-914 the lower-bound term is `(idxValue >= 0 || scatterUpdateMode == ScatterElementsUpdate)`, which for ScatterElementsUpdate is unconditionally true, so no negative lower bound is enforced. Inside scatterElementsUpdate the normalization `if (idxValue<0) idxValue += data_dim_size` (lines 602-603,624-625,647-648,666-667) only fixes indices in [-d,-1]; an index like -1000 with d=10 stays at -990. The subsequent `assert(idxValue < data_dim_size && idxValue >= 0)` (605,627,650,669,734,758,793,814) is a no-op in release/NDEBUG builds. The address computation `dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]` (628 etc.) multiplies int64_t(-990) by size_t dataBlock_axisplus1 (confirmed size_t at line 712); usual arithmetic conversion promotes the negative int64_t to a huge uint64_t, yielding an out-of-bounds destination written via kernel(dst,src). vulnType CWE-129 leading to CWE-787 is accurate; impact (controlled heap OOB write under attacker model) is accurate. The proposed fix (add `idxValue >= -srcDimAxis` for ScatterElementsUpdate) is correct in direction but INSUFFICIENT alone: the CPU_NODE_ASSERT at 913 lives inside the `if (axisRelaxed)` branch (line 890), so when axis is a compile-time constant (axisRelaxed==false) that loop never runs and no bounds check exists at all. The essential part of the fix is the finding's second recommendation: replace the eight release-stripped `assert(idxValue < data_dim_size && idxValue >= 0)` calls with CPU_NODE_ASSERT (or clamp/reject) inside scatterElementsUpdate itself so the lower bound is enforced unconditionally in release builds at the point of use. Doing both is sufficient.

## Exploit / Proof of Concept
1. Craft a model with a ScatterElementsUpdate node whose INDICES tensor on axis 0 contains the value -1000, while the data tensor's axis-0 dimension is 10. 2. CPU_NODE_ASSERT (line 913) passes: `-1000 < 10` is true, and `(false || true)` is true for ScatterElementsUpdate mode. 3. Inside scatterElementsUpdate<DataType,KernelType>() the correction at e.g. line 624 produces `idxValue = -1000 + 10 = -990`. 4. `assert(idxValue >= 0)` at line 627 is stripped in release builds. 5. The expression `offsets[0] + idxValue * dataBlock_axisplus1` at line 628 promotes int64_t(-990) to uint64_t(≈0xFFFFFFFFFFFFFC22), multiplies by block size, and adds to a valid base offset, yielding a wildly out-of-bounds destination pointer used in `kernel(dst, src)` — a controlled OOB write.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-129/787 at openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:913-628.
// Pre-fix: a ScatterElementsUpdate node whose INDICES contain a value < -data_dim_size
// (e.g. -1000 on axis-0 size 10) passes the CPU_NODE_ASSERT at line 913 (OR-clause) and the
// release-stripped assert at line 627, then computes dataPtr[offsets[0] + int64_t(-990)*block]
// where the negative int64_t is promoted to a huge uint64_t => heap OOB write (ASan: heap-buffer-overflow WRITE).
// Post-fix: the node must reject/throw on out-of-range negative indices, OR enforce idxValue>=-data_dim_size.
//
// SKELETON: the exact intel_cpu unit-test fixture symbols (ov_cpu_unit_tests node-test helpers)
// must be confirmed against intel_cpu/tests/unit before this will compile.
#include <gtest/gtest.h>
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/runtime/core.hpp"

// TODO: confirm correct test target & include layout under
//       openvino/src/plugins/intel_cpu/tests/unit/ — symbol names below are placeholders.
TEST(ScatterElementsUpdateCpu, RejectsDeeplyNegativeIndex) {
    using namespace ov;
    // data: shape [10], axis 0; indices: [-1000]; updates: [42]
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto indices = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1000});
    auto updates = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{42.0f});
    auto axis    = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    Tensor in(element::f32, Shape{10});
    std::fill_n(in.data<float>(), 10, 0.0f);
    req.set_input_tensor(in);
    // Pre-fix this performs an OOB write (ASan abort). Post-fix the node must reject the index.
    ASSERT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests, ASan-enabled config). Run: ov_cpu_unit_tests --gtest_filter=ScatterElementsUpdateCpu.RejectsDeeplyNegativeIndex . Expected pre-fix: AddressSanitizer 'heap-buffer-overflow WRITE' inside ScatterUpdate::scatterElementsUpdate (dataPtr[offsets[0] + idxValue*dataBlock_axisplus1], scatter_update.cpp:628). Expected post-fix: index rejected (CPU_NODE_ASSERT/throw) and ASSERT_ANY_THROW passes with no ASan report. TODO: replace placeholder fixture symbols with the real intel_cpu node-test helpers found under intel_cpu/tests/unit/.

## Suggested fix
In execute() at line 913, add the missing lower-bound constraint for ScatterElementsUpdate: `CPU_NODE_ASSERT(idxValue < static_cast<int64_t>(srcDimAxis) && (idxValue >= -static_cast<int64_t>(srcDimAxis) || scatterUpdateMode != ScatterElementsUpdate) && (idxValue >= 0 || scatterUpdateMode == ScatterElementsUpdate), ...)`. Equivalently, when `scatterUpdateMode == ScatterElementsUpdate`, require `idxValue >= -static_cast<int64_t>(srcDimAxis)` in addition to the existing upper-bound check. Additionally, replace all eight `assert(idxValue < data_dim_size && idxValue >= 0)` calls (lines 605, 627, 650, 669, 734, 758, 793, 814) with CPU_NODE_ASSERT so they are enforced in release builds as defence-in-depth.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #502.
