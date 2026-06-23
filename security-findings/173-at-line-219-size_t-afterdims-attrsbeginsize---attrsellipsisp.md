# Security finding #173: At line 219, `size_t afterDims = attrs.begin.size() - attrs.ellipsi…

**Summary:** At line 219, `size_t afterDims = attrs.begin.size() - attrs.ellipsi…

**CWE IDs:** CWE-191: Integer Underflow (Unsigned Wrapping) leading to CWE-787: Out-of-bounds Write / CWE-400: Uncontrolled Resource Consumption
**Severity / Impact:** Unbounded `push_back` loop (line 228-230) causes uncontrolled heap consumption / OOM crash, or — on allocators that reuse freed memory — heap corruption. Triggered at model-loading time (initSupportedPrimitiveDescriptors, line 306) with no privilege required beyond supplying a malformed model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/strided_slice.cpp:219` — `addHiddenDims (static)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied model file → ov::op::v1::StridedSlice ellipsis_mask and begin attributes → attrs.ellipsisPos1 and attrs.begin.size() → addHiddenDims

## Description / Root cause
At line 219, `size_t afterDims = attrs.begin.size() - attrs.ellipsisPos1 - 1` performs an unsigned subtraction where `attrs.ellipsisPos1` (int, set to the index of the ellipsis-mask bit) is implicitly widened to `size_t`. If `attrs.ellipsisPos1 >= (int)attrs.begin.size()`, the subtraction wraps around to a very large `size_t`. At line 220, `ellipsisPos2 = inputRank - afterDims - 1` then also underflows (huge `afterDims` > `inputRank`). At line 228, the loop `for (size_t i = attrs.ellipsisPos1; i < ellipsisPos2 + 1; i++) temp.push_back(bit)` iterates up to ~2^64 times, causing unbounded heap growth. The guard at line 215 only checks `inputRank > 3 && attrs.equalDims && attrs.ellipsisMaskCounter == 1`; it does NOT verify `(size_t)attrs.ellipsisPos1 < attrs.begin.size()`.

**Validator analysis:** The underflow is genuine and reachable: ellipsisMask is padded to nDims (strided_slice.cpp:123) so ellipsisPos1 (line 151) can be >= begin.size(), while begin is deliberately NOT padded when an ellipsis is present (line 172 gates padding on ellipsisMaskCounter==0). Neither the op-level validation (core/src/op/strided_slice.cpp:122-140, which only checks masks are 0/1, masks share one size, and <=1 ellipsis) nor shape inference (strided_slice_shape_inference.hpp:126-148, which iterates axis<number_axes==begin.size() and so never visits an ellipsis bit positioned beyond begin) ties ellipsisPos1 to begin.size(); the malformed IR therefore loads and reaches addHiddenDims via initSupportedPrimitiveDescriptors (line 306). The CWE-191 underflow classification is accurate. However the stated 'OOB write' is imprecise: the primary memory-safety effect is an OOB READ at line 226 (data[i]) and line 232 (data[i+ellipsisPos1]) plus an unbounded/underflowed push_back loop at lines 228 and 231 (inputRank-ellipsisPos2 also underflows) — i.e. CWE-125 OOB read + CWE-400 resource exhaustion rather than CWE-787. The proposed fix is correct and necessary but not fully sufficient: it should also guard line 231's `inputRank - ellipsisPos2` (validate ellipsisPos2 < inputRank) and ensure ellipsisPos2 >= ellipsisPos1, and bound the data[] indexing against begin.size(); ideally add a NODE_VALIDATION_CHECK at op level tying the ellipsis position to the begin/end length so all consumers are protected, not just the CPU node.

## Exploit / Proof of Concept
Craft an ONNX/IR model containing an ov::op::v1::StridedSlice node where the `ellipsis_mask` vector has length N (e.g. N=4), with bit position 3 set, but the `begin` constant tensor has shape [1] (size 1). `ellipsisPos1` is set to 3 at line 151, `attrs.begin.size()` is 1. At line 219: `1 - 3 - 1` in `size_t` arithmetic wraps to `SIZE_MAX - 2` (~2^64-3). At line 220: `inputRank(e.g.4) - (SIZE_MAX-2) - 1` wraps to a small but incorrect value, or itself underflows further. The loop at line 228 then runs for an astronomically large number of iterations, exhausting heap memory and crashing or corrupting the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-191 unsigned underflow in addHiddenDims()
// at openvino/src/plugins/intel_cpu/src/nodes/strided_slice.cpp:219-220.
//
// Pre-fix: building a v1::StridedSlice with ellipsis_mask whose set bit index
// (ellipsisPos1) is >= begin.size() makes
//     afterDims  = begin.size() - ellipsisPos1 - 1   // size_t underflow -> ~2^64
//     ellipsisPos2 = inputRank - afterDims - 1        // further underflow
// and the loops at lines 226/228/231/232 read out of bounds / push_back
// unbounded, detected by ASan as a heap-buffer-overflow or OOM.
// Post-fix: the added bounds check (ellipsisPos1 >= begin.size() ->
// OPENVINO_THROW / NODE_VALIDATION_CHECK) makes graph compile reject the model.
//
// Harness: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit/.
//
// SKELETON — addHiddenDims is a file-static function and cannot be called
// directly; it is only reachable through StridedSlice::initSupportedPrimitiveDescriptors().
// The exact node-construction / Graph-compile helper symbols must be copied from
// an existing intel_cpu/tests/unit node test before this will compile.

#include <gtest/gtest.h>
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(StridedSliceCpuNode, EllipsisPosBeyondBeginSizeIsRejected) {
    // input data rank 4 (>3, required to enter the line-215 branch)
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});
    // begin/end/stride constants of length 1 -> begin.size() == 1
    auto begin  = op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto end    = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto stride = op::v0::Constant::create(element::i64, Shape{1}, {1});

    // ellipsis_mask length 4 with the set bit at index 3 -> ellipsisPos1 == 3,
    // which is >= begin.size() (==1): triggers the size_t underflow at line 219.
    const std::vector<int64_t> begin_mask{0, 0, 0, 0};
    const std::vector<int64_t> end_mask{0, 0, 0, 0};
    const std::vector<int64_t> new_axis_mask{0, 0, 0, 0};
    const std::vector<int64_t> shrink_axis_mask{0, 0, 0, 0};
    const std::vector<int64_t> ellipsis_mask{0, 0, 0, 1};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride,
                                                     begin_mask, end_mask,
                                                     new_axis_mask, shrink_axis_mask,
                                                     ellipsis_mask);
    auto model = std::make_shared<Model>(OutputVector{ss->output(0)}, ParameterVector{data});

    // TODO: replace with the intel_cpu unit-test graph-compile helper that drives
    //       StridedSlice::initSupportedPrimitiveDescriptors() (the entry that calls
    //       addHiddenDims at strided_slice.cpp:306). Copy the exact helper /
    //       fixture symbol from an existing test under intel_cpu/tests/unit/ (e.g.
    //       a Graph/Node builder). Pre-fix this aborts under ASan; post-fix it must
    //       throw ov::Exception.
    // EXPECT_THROW(compile_cpu_node_supported_descriptors(model), ov::Exception);
    GTEST_SKIP() << "TODO: wire to intel_cpu node compile helper to reach addHiddenDims";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests; Run: ./ov_cpu_unit_tests --gtest_filter=StridedSliceCpuNode.EllipsisPosBeyondBeginSizeIsRejected ; Expected pre-fix: ASan 'heap-buffer-overflow READ' at strided_slice.cpp:226/232 or 'out-of-memory' from the unbounded push_back loop (lines 228/231); Expected post-fix: ov::Exception thrown (graph compile rejects the model).

## Suggested fix
Before the subtraction at line 219, add a bounds check: `if (static_cast<size_t>(attrs.ellipsisPos1) >= attrs.begin.size()) OPENVINO_THROW("StridedSlice: ellipsisPos1 (", attrs.ellipsisPos1, ") is out of range for begin.size()=", attrs.begin.size());`. Additionally, after computing `afterDims`, validate `afterDims < inputRank` before computing `ellipsisPos2` at line 220 to prevent a second underflow: `if (afterDims >= inputRank) OPENVINO_THROW("StridedSlice: afterDims underflow");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #173.
