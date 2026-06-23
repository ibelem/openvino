# Security finding #141: When reverseIndexing==true, line 982 does `ii += axisDim` (int32_t …

**Summary:** When reverseIndexing==true, line 982 does `ii += axisDim` (int32_t …

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Heap out-of-bounds read of up to ~380 bytes (for ii=-95, reading 95×4 bytes before the allocation). Can leak heap metadata, adjacent tensor contents, or pointers, enabling info-leak and potentially assisting heap exploitation.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:982` — `Gather::exec1DCase()`
**Validated for repos:** openvino
**Trust boundary:** Indices tensor values from a deserialized model (v8::Gather without dontReverseIndices, or GatherCompressed) flow into exec1DCase where a very-negative index survives the reverseIndexing adjustment

## Description / Root cause
When reverseIndexing==true, line 982 does `ii += axisDim` (int32_t += size_t). If the attacker supplies an index more negative than -(axisDim) (e.g. -100 with axisDim=5), after the adjustment ii is still negative (-95). There is no lower-bound check before psrc[ii] on line 987. A negative int32_t subscript converts to a large negative ptrdiff_t, reading far before the start of the data buffer. execReference is similarly unguarded for this lower-bound case, but exec1DCase is the fast-path used for the small 1-D tensors common in shape-inference subgraphs.

**Validator analysis:** The defect is real. In Gather::exec1DCase the index value `ii` is read as int32_t (line 979) from the runtime indices tensor; for negative indices with reverseIndexing==true the only adjustment is `ii += axisDim` (line 982). Because axisDim is size_t the addition is done in unsigned then truncated back to int32_t, so an index more negative than -axisDim (e.g. -100 with axisDim=5) stays negative (-95). Line 987 then does `psrc[ii]` with no lower-bound check, producing a negative ptrdiff_t subscript and an out-of-bounds read before the data buffer. Note exec1DCase is *worse* than claimed: it also lacks the upper-bound guard present in execReference (line 947 `if (idx < axisDim)`), so any positive `ii >= axisDim` and the !reverseIndexing fallback `ii = axisDim` (line 984) also read out of bounds — so CWE-125 (Out-of-bounds Read) is the correct category and the heap-info-leak impact is accurate. Reachability is confirmed: prepareParams sets canOptimize1DCase for i32 1-D data/indices with dim<=64 (lines 396-401), and reverseIndexing is true for a v8::Gather lacking the dontReverseIndices rt_info (line 115) or for GatherCompressed (line 121). The proposed fix is correct AND sufficient: inserting `if (ii < 0 || static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` after line 982 supplies both the missing lower bound (post-adjustment negative) and the missing upper bound, matching execReference's idx<axisDim semantics and also covering the !reverseIndexing ii=axisDim case. For openvinoEp the finding is marked na: the cited code and the flawed index arithmetic exist entirely within OpenVINO's CPU plugin runtime, not in the EP's plugin_impl, which performs no Gather index handling; claiming EP reachability would require unproven assumptions (device==CPU, subgraph not constant-folded, runtime negative-index input) that the cited code does not establish.

## Exploit / Proof of Concept
Craft a model with an ov::op::v8::Gather node (no dontReverseIndices → reverseIndexing=true), 1-D int32 data of shape [5], 1-D indices of shape [1] containing value -100. prepareParams enables canOptimize1DCase (line 401). exec1DCase: ii=-100, ii<0, ii+=5 → ii=-95, psrc[-95] reads 95 elements (380 bytes) before the buffer start.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in Gather::exec1DCase (openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:982-987).
// Pre-fix: indices value -100 with 1-D i32 data of shape [5] survives `ii += axisDim` as ii=-95,
//          then `psrc[ii]` (line 987) reads ~380 bytes before the data buffer -> ASan heap-buffer-overflow (read).
// Post-fix: the added `if (ii < 0 || (size_t)ii >= axisDim)` guard zero-fills the output, no OOB.
// Harness: ov_cpu_unit_tests (gtest + ASan). Place near intel_cpu/tests/unit/ Gather node tests.
//
// SKELETON — the exact intel_cpu unit-test fixture/helpers for constructing and executing a single
// Gather node graph were not read, so symbol names below are placeholders.
#include <gtest/gtest.h>
// TODO: include the intel_cpu unit-test graph/node helpers actually used under
//       src/plugins/intel_cpu/tests/unit/ (e.g. the test util that builds a CPU graph from an ov::Model
//       and runs infer). Read that tree to get the real headers and runner symbols.

#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(intel_cpu_Gather, exec1DCase_negative_index_below_minus_axisdim_no_oob) {
    // 1-D i32 data of shape [5], 1-D i32 indices of shape [1] = {-100}.
    // dims<=64 and i32 => prepareParams enables canOptimize1DCase (gather.cpp:399-401),
    // and v8::Gather without dontReverseIndices => reverseIndexing==true (gather.cpp:115).
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{-100});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims=*/0);
    auto model   = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    // TODO: build the CPU graph for `model` and run infer with input data {0,1,2,3,4}
    //       using the intel_cpu unit-test runner. With the fix the single output element
    //       must be 0 (out-of-range index -> zero-fill, matching execReference at line 947-963).
    //
    //   auto out = run_cpu_single_infer(model, /*input i32[5]=*/{0,1,2,3,4});
    //   ASSERT_EQ(out.size(), 1u);
    //   EXPECT_EQ(out[0], 0);   // fix zero-fills; pre-fix this line is never reached (ASan abort earlier)
    GTEST_SKIP() << "TODO: wire intel_cpu unit-test graph runner; see gather.cpp:982-987";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter='intel_cpu_Gather.exec1DCase_negative_index_below_minus_axisdim_no_oob'. Expected pre-fix: ASan 'heap-buffer-overflow READ of size 4' originating at Gather::exec1DCase psrc[ii] (gather.cpp:987), ~380 bytes before the data allocation. Expected post-fix: test passes (output element == 0). NOTE: skeleton — complete the CPU single-infer runner wiring (see TODOs) before building.

## Suggested fix
After the adjustment at line 982, check `if (ii < 0 || static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` before line 987. This adds both a lower-bound (ii<0 after adjustment) and upper-bound check, matching the semantics of execReference lines 947-963.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #141.
