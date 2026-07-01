# Security finding #17: At lines 605, 627, 650, and 669, the only bounds check after negati…

**Summary:** At lines 605, 627, 650, and 669, the only bounds check after negati…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Out-of-bounds write with attacker-controlled offset and value (update tensor contents). In a release build an adversary providing a crafted ONNX/OpenVINO model or inference-time index tensor can corrupt heap memory at an arbitrary offset below the data buffer, leading to potential remote code execution or denial of service. Any application embedding the OpenVINO CPU inference engine and running ScatterElementsUpdate is affected.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:605` — `ScatterUpdate::scatterElementsUpdate<DataType,KernelType>()`
**Validated for repos:** openvino
**Trust boundary:** User-supplied indices tensor at INDICES_ID port, deserialized from model or provided at inference time

## Description / Root cause
At lines 605, 627, 650, and 669, the only bounds check after negative-index normalization is `assert(idxValue < data_dim_size && idxValue >= 0)`. In release builds this assert is stripped (NDEBUG). The pre-dispatch validation loop (lines 913-914) explicitly skips the `idxValue >= 0` check for ScatterElementsUpdate mode, so a very-negative index such as -1000 for a dim-10 axis passes validation, is normalized to -990 (still negative), and reaches `dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]` (e.g. line 606) where the signed-negative product is implicitly converted to a huge unsigned offset via size_t arithmetic, causing an out-of-bounds write arbitrarily far below the data buffer.

**Validator analysis:** vulnType CWE-787 (Out-of-bounds Write) is accurate: the sink dataPtr[offsets[0] + idxValue * dataBlock_axisplus1] (lines 606/628/651/670, and 735/759/794/815 in the ReduceMean specialization) writes attacker-controlled data at an attacker-influenced offset. The defect is real: (1) assert() is a no-op under NDEBUG release builds; (2) normalization at 602-603/624-625/647-648/666-667 adds data_dim_size exactly once, so any index < -data_dim_size stays negative; (3) the only release-active validation, lines 913-914, both lives behind `if (axisRelaxed)` (line 890) AND explicitly OR-exempts `idxValue >= 0` for ScatterElementsUpdate, so out-of-range negatives are never rejected. Because idxValue is int64_t and dataBlock_axisplus1 is size_t, the negative product is converted to a large unsigned value, addressing memory far outside the buffer. Note: the loop at 911-916 iterates indicesBlockND[0] (=total indices count) which is correct for full coverage, but its condition is the flaw, not its range. The impact statement (heap corruption, potential RCE/DoS) is plausible for a release build, though the write value is the update tensor element (semi-controlled) and the base offset is data-buffer-relative, so 'arbitrary offset below the data buffer' is the precise characterization. The proposedFix is correct and sufficient in principle: replacing the bare asserts with CPU_NODE_ASSERT(idxValue >= 0 && idxValue < data_dim_size, ...) converts the violation into a thrown ov::Exception at the API boundary. A cleaner/cheaper alternative is to fix the pre-dispatch loop (913-914) to also enforce non-negativity for ScatterElementsUpdate after normalization AND to run that validation unconditionally (not only under axisRelaxed) — but note the per-element asserts must still be hardened because the normalization currently happens per-element and the loop at 911 reads raw (un-normalized) indices; the safest minimal fix is the CPU_NODE_ASSERT replacement at every per-element sink (8 sites), which guarantees correctness regardless of axisRelaxed. Recommend the CPU_NODE_ASSERT replacement as the primary fix.

## Exploit / Proof of Concept
Supply an index tensor containing a value whose absolute value exceeds the axis dimension size (e.g., index = -1000 for data axis dim = 10). The pre-dispatch check at line 913-914 passes because the `idxValue >= 0` sub-condition is OR-ed away for ScatterElementsUpdate mode. Normalization at line 603 produces -990, which remains negative. In a release build the assert at line 605 is absent (NDEBUG). The expression `offsets[0] + (-990LL) * dataBlock_axisplus1` involves an implicit conversion of a large negative int64 product to size_t, yielding an address far below `dataPtr`. The kernel then writes the attacker-controlled update value to that address, corrupting heap metadata or other allocations.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-787 OOB write in
//   openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:602-670
// (and the ReduceMean specialization at 700-815).
//
// Encodes the fix: a ScatterElementsUpdate whose indices tensor contains a
// value more negative than -data_dim_size (e.g. -1000 for an axis of size 10)
// must be REJECTED with an ov::Exception (via CPU_NODE_ASSERT) instead of
// performing an out-of-bounds write. Pre-fix, in an NDEBUG/release build the
// bare assert() at line 605/627/650/669 is stripped and the negative index
// (normalized once to -990, still negative) flows into
//   dataPtr[offsets[0] + idxValue * dataBlock_axisplus1]
// producing a wild store that ASan reports as heap-buffer-overflow.
//
// NOTE (why this is a SKELETON, not a turnkey test):
//  * The OOB only manifests under NDEBUG; a Debug-built ov_cpu_unit_tests will
//    instead abort on the surviving assert(), so the assertion semantics differ
//    by build type. The fix replaces assert() with CPU_NODE_ASSERT, which throws
//    in BOTH build types — that is what we assert below.
//  * The exact node-construction / infer-request helper symbols for driving a
//    single ScatterElementsUpdate op through ov_cpu_unit_tests must be copied
//    from the surrounding nodes/ test tree (e.g. the Subgraph/ngram node tests);
//    fill the TODOs after reading them.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_elements_update.hpp"

using namespace ov;

// TODO: confirm the fixture/helper used by neighbouring intel_cpu/tests/unit/nodes/*.cpp
// for compiling and infering a tiny single-op model on the CPU plugin.
TEST(ScatterElementsUpdateCPU, NegativeOutOfRangeIndexIsRejected) {
    // data: shape [10], axis 0  => data_dim_size = 10
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    auto seu = std::make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{seu},
                                         ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<float> data_vals(10, 0.0f);
    std::vector<int32_t> idx_vals{-1000};  // |idx| >> data_dim_size, stays negative after +10
    std::vector<float> upd_vals{42.0f};

    req.set_input_tensor(0, Tensor(element::f32, Shape{10}, data_vals.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{1}, idx_vals.data()));
    req.set_input_tensor(2, Tensor(element::f32, Shape{1}, upd_vals.data()));

    // Post-fix: CPU_NODE_ASSERT converts the out-of-range index into a thrown
    // exception. Pre-fix (release/NDEBUG): silent OOB write (ASan heap-buffer-overflow).
    EXPECT_THROW(req.infer(), ov::Exception);
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run: ./ov_cpu_unit_tests --gtest_filter=ScatterElementsUpdateCPU.NegativeOutOfRangeIndexIsRejected. Expected: PRE-FIX in a release+ASan build, ASan reports 'heap-buffer-overflow WRITE' originating from ScatterUpdate::scatterElementsUpdate at scatter_update.cpp:606 (dataPtr[offsets[0] + idxValue*dataBlock_axisplus1]); the EXPECT_THROW fails because no exception is thrown. POST-FIX, CPU_NODE_ASSERT throws ov::Exception and the test passes with no ASan report. TODO: replace the model-driving boilerplate with the exact helper used by neighbouring intel_cpu/tests/unit/nodes tests after reading them.

## Suggested fix
Replace all four bare `assert(idxValue < data_dim_size && idxValue >= 0)` checks (lines 605, 627, 650, 669 in scatterElementsUpdate<DataType,KernelType> and lines 734, 758, 793, 814 in the ReduceMean specialization) with a runtime enforcement such as `CPU_NODE_ASSERT(idxValue >= 0 && idxValue < data_dim_size, "Index out of bounds in ScatterElementsUpdate");`. Additionally, tighten the pre-dispatch validation loop at line 914 to also reject negative indices for ScatterElementsUpdate (remove the `|| scatterUpdateMode == ScatterElementsUpdate` exemption), or perform the negative-index normalization there and then assert the result is non-negative, so the per-element sinks are never reached with an out-of-range value.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #17.
