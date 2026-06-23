# Security finding #74: The `execute()` pre-check (lines 913–914) intentionally allows any …

**Summary:** The `execute()` pre-check (lines 913–914) intentionally allows any …

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Arbitrary heap write. An attacker who can supply a crafted model (or crafted indices tensor at inference time) can corrupt heap memory beyond the allocated data buffer, leading to denial of service (crash) or potentially remote code execution.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:605` — `ScatterUpdate::scatterElementsUpdate<DataType,KernelType>()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX/IR model supplies the indices tensor values, which flow through the OpenVINO EP into the CPU plugin kernel write sites

## Description / Root cause
The `execute()` pre-check (lines 913–914) intentionally allows any negative `idxValue` for `ScatterElementsUpdate` mode — it only asserts `idxValue < srcDimAxis` with no lower-bound check. Inside `scatterElementsUpdate`, normalization `if (idxValue < 0) { idxValue += data_dim_size; }` is the only correction, followed solely by `assert(idxValue < data_dim_size && idxValue >= 0)` (lines 605, 627, 650, 669 and their ReduceMean counterparts 734, 758, 793, 814). In a release build those asserts are compiled away. A value like `-(data_dim_size + 1)` survives the pre-check (it is less than `srcDimAxis`), normalization yields `-1`, and the expression `dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]` converts the negative `int64_t` product to a large `size_t`, producing an out-of-bounds heap write at an attacker-controlled offset.

**Validator analysis:** The CWE-787 out-of-bounds write classification is accurate. I traced the trust boundary: untrusted indices tensor → ScatterUpdate::execute() (line 852) → the relaxed-axis pre-check at lines 913-915 → scatterElementsUpdate<DataType,KernelType> (line 555). The pre-check's `(idxValue >= 0 || scatterUpdateMode == ScatterElementsUpdate)` clause deliberately lets ANY negative index pass for ScatterElementsUpdate, with only an upper-bound `idxValue < srcDimAxis` enforced. The kernel then normalizes once (`idxValue += data_dim_size`), so an index of e.g. `-(data_dim_size+1)` becomes `-1` (still negative), and `idxValue * dataBlock_axisplus1` (int64 * size_t) yields a negative/huge offset producing a write before/far from the buffer. The only correctness guard at the eight cited lines is plain `assert()`, which is a no-op in release (NDEBUG) builds — so the flaw is real and reachable in shipped builds. Note the pre-check at 911-916 runs only when `axisRelaxed` is true (axis supplied as a runtime input port); when axis is a static attribute, the pre-check loop is skipped entirely, making the kernel-side asserts the ONLY guard — strengthening the finding. The proposed fix is correct in direction but INSUFFICIENT/imprecise: (1) enforcing `idxValue >= -srcDimAxis` at execute() only covers the axisRelaxed path; the non-relaxed (static axis) path bypasses 911-916, so the real fix MUST be in the kernel. (2) Better fix: replace each `assert(idxValue < data_dim_size && idxValue >= 0)` with an active `CPU_NODE_ASSERT(idxValue >= 0 && idxValue < data_dim_size, "scatter index out of range")` AFTER normalization at all eight sites (605/627/650/669/734/758/793/814) — that catches both the unnormalizable < -data_dim_size case and any residual negative, in both axis paths and in release builds. Tightening the execute() pre-check to `[-srcDimAxis, srcDimAxis-1]` is good defense-in-depth but not sufficient alone.

## Exploit / Proof of Concept
Craft an ONNX model with a ScatterElementsUpdate node whose indices tensor contains a value of `-(data_dim_size + 1)` on the target axis. At execute() (line 913–914), `idxValue < srcDimAxis` is true (a large negative is less than any positive dimension), and the ScatterElementsUpdate exemption allows it past the lower-bound check. Inside scatterElementsUpdate, `idxValue += data_dim_size` gives `-1`. The release-build assert is a no-op. The subscript `dataPtr[offsets[0] + (-1LL) * dataBlock_axisplus1]` promotes the negative int64 product to a large size_t, writing to memory well before the allocated buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp
//   - exemption at lines 913-915 (execute pre-check skips lower-bound for ScatterElementsUpdate)
//   - single-step normalization 602-603/624-625/647-648/666-667/731-732/755-756/790-791/811-812
//   - dead asserts at 605/627/650/669/734/758/793/814 (no-op under NDEBUG)
//
// Intent: feed a ScatterElementsUpdate op an index value of -(data_dim_size+1)
// on the scatter axis. Pre-fix: index normalizes to -1, the dead assert is a
// no-op in release, and dataPtr[offsets[0] + (-1)*dataBlock_axisplus1] writes
// out of bounds (ASan: heap-buffer-overflow WRITE). Post-fix: an active
// CPU_NODE_ASSERT after normalization rejects the index -> ov::Exception.
//
// TODO(verify): exact target/symbols. File lives under intel_cpu, so the
// harness is `ov_cpu_unit_tests`. This is a SKELETON: the direct node API
// (ScatterUpdate node) is not trivially constructible standalone, so the test
// drives it via an ov::Model + CPU plugin infer request, which is the pattern
// used by intel_cpu single-layer tests. Confirm helper/include names against
// the existing intel_cpu test tree before use.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// TODO: confirm CPU device is registered as "CPU" in this test target's setup.
TEST(scatter_elements_update_cpu, negative_index_below_neg_dim_is_rejected) {
    // data: shape [4] -> data_dim_size on axis 0 is 4
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});

    // indices: a single out-of-range negative value -(4+1) = -5 on axis 0.
    // Pre-fix normalization gives -5 + 4 = -1 (still negative) -> OOB write.
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-5});
    auto updates = op::v0::Constant::create(element::f32, Shape{1}, {42.0f});
    auto axis    = op::v0::Constant::create(element::i32, Shape{1}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor input(element::f32, Shape{4});
    auto* p = input.data<float>();
    for (size_t i = 0; i < 4; ++i) p[i] = 1.0f;
    req.set_input_tensor(input);

    // Post-fix: kernel rejects the unnormalizable negative index.
    // Pre-fix: ASan flags heap-buffer-overflow WRITE inside scatterElementsUpdate.
    EXPECT_ANY_THROW(req.infer());
}
```
**Build / run:** Build target: ov_cpu_unit_tests (OpenVINO built with -DENABLE_SANITIZER=ON / ASan). Run: ./ov_cpu_unit_tests --gtest_filter=scatter_elements_update_cpu.negative_index_below_neg_dim_is_rejected . Expected pre-fix: AddressSanitizer 'heap-buffer-overflow WRITE' originating in ScatterUpdate::scatterElementsUpdate (scatter_update.cpp ~line 606/628/651/670) because idxValue normalizes to -1 and the dead assert(605...) is a no-op in release. Expected post-fix: infer() throws ov::Exception from the new active CPU_NODE_ASSERT, test passes. NOTE: skeleton — verify exact CPU test include paths/device name against intel_cpu/tests/unit before relying on it.

## Suggested fix
In `ScatterUpdate::execute()`, remove the lower-bound exemption for `ScatterElementsUpdate` mode and enforce `idxValue >= -static_cast<int64_t>(srcDimAxis)` for all modes. Change line 913–914 to: `CPU_NODE_ASSERT(idxValue < static_cast<int64_t>(srcDimAxis) && idxValue >= -static_cast<int64_t>(srcDimAxis), "indices value out of valid range [-srcDimAxis, srcDimAxis-1]");`. Also replace the `assert()` calls at lines 605, 627, 650, 669, 734, 758, 793, 814 with `CPU_NODE_ASSERT(idxValue >= 0 && idxValue < data_dim_size, ...)` so the check is active in release builds as a defense-in-depth measure.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #74.
