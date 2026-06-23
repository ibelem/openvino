# Security finding #140: When reverseIndexing==false (set unconditionally for v7::Gather at …

**Summary:** When reverseIndexing==false (set unconditionally for v7::Gather at …

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Attacker-controlled negative index causes a 4-byte out-of-bounds read from the data buffer. Under ASLR this leaks heap metadata or adjacent allocation contents (info-leak). If the adjacent memory is also written by a race or earlier step, it could be used to leak pointers and aid further exploitation. Affects any deployment loading v7::Gather or v8::Gather+dontReverseIndices models with 1-D int32 data and index tensors whose dimensions are ≤64 (the canOptimize1DCase fast path at lines 396-403).
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvino
**Trust boundary:** Indices tensor values supplied from a deserialized model file (v7::Gather or v8::Gather with dontReverseIndices rt_info) flow into psrc[] with no upper-bound guard

## Description / Root cause
When reverseIndexing==false (set unconditionally for v7::Gather at line 118, and for v8::Gather with dontReverseIndices at line 115), a negative index sets ii=axisDim (line 984, the sentinel). Line 987 then executes psrc[axisDim] — one element past the end of the data buffer — with no guard. execReference avoids this via `if (idx < static_cast<size_t>(axisDim))` at line 947, but exec1DCase has no equivalent check.

**Validator analysis:** Confirmed. exec1DCase (gather.cpp:967-989) reads `pdst[i]=psrc[ii]` at line 987 with no upper-bound check. For v7::Gather reverseIndexing is unconditionally false (line 118), and for v8::Gather+dontReverseIndices it is false (line 115); a negative index then sets ii=axisDim (the sentinel, line 984) and line 987 reads psrc[axisDim] — one uint32_t past the data buffer. Note the bug is actually broader than the report: there is ALSO no guard for positive out-of-range indices (e.g. idx=100 with axisDim=4), which exec1DCase would happily dereference too — execReference handles both via `if (idx < axisDim)` at line 947 (else memset 0). The path is reachable: prepareParams sets canOptimize1DCase=true for 1-D i32 data/indices with dims ≤64 (lines 396-403), and execute() dispatches to exec1DCase at line 466-468. CWE-125 Out-of-bounds Read and the 4-byte info-leak impact are accurate. The proposed fix `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` is correct and sufficient — it mirrors execReference and also covers the positive-overflow case (since size_t cast makes the comparison cover both). Recommend placing the guard right after the ii-adjustment block (after line 985). openvinoEp is na: the defect is CPU-plugin internal and not present in plugin_impl; though a malicious model could in principle traverse ORT→OpenVINO-EP→CPU, the narrow 1-D int32 fast path is an OpenVINO-internal concern and the EP code holds no part of it.

## Exploit / Proof of Concept
Craft a model with an ov::op::v7::Gather node (reverseIndexing=false), 1-D int32 data tensor of size ≤64 (e.g. shape [4]), and 1-D indices tensor of size ≤64 containing a single value -1. prepareParams sets canOptimize1DCase=true (line 401). At runtime exec1DCase is called; ii=-1 < 0 triggers the else-branch at line 984, setting ii=axisDim=4. psrc[4] reads 4 bytes past the end of a 16-byte (4-element) buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read in Gather::exec1DCase (gather.cpp:987).
// Pre-fix: with a v7::Gather (reverseIndexing==false), 1-D int32 data of size 4
// and a 1-D int32 indices tensor containing a negative value (-1), exec1DCase
// sets ii=axisDim=4 (line 984) and reads psrc[4] — 4 bytes past the buffer end.
// ASan reports a heap-buffer-overflow READ. Once the fix adds the
// `if ((size_t)ii >= axisDim) { pdst[i]=0; continue; }` guard, the read is
// suppressed and the out-of-range lane is zeroed.
//
// TODO: confirm exact target/harness by reading
//       openvino/src/plugins/intel_cpu/tests/unit/ (target ov_cpu_unit_tests)
//       and the single-layer-test helpers used for Gather there.
// TODO: the 1-D int32 fast path (canOptimize1DCase) requires dataSrcRank<=1,
//       i32 precision, and dims<=64 — build the ov::Model accordingly.

#include <gtest/gtest.h>
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"
// TODO: include the intel_cpu unit-test fixture headers that let you compile
//       and infer a model on the CPU plugin (see tests/unit/ examples).

TEST(GatherCpu1DCase, NegativeIndexNoOobRead) {
    using namespace ov;
    // 1-D i32 data, size 4  -> axisDim = 4, qualifies for canOptimize1DCase
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // 1-D i32 indices, single value -1 (out-of-range / sentinel trigger)
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-1});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather = std::make_shared<op::v7::Gather>(data, indices, axis); // v7 => reverseIndexing=false
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    // TODO: compile `model` on the CPU plugin and run inference with a 4-element
    //       int32 input. Pre-fix this triggers an ASan heap-buffer-overflow READ
    //       inside Gather::exec1DCase at gather.cpp:987. Post-fix the output lane
    //       for the -1 index must be 0 and no OOB access occurs.
    //   ov::Core core; auto cm = core.compile_model(model, "CPU");
    //   auto req = cm.create_infer_request();
    //   ... set input {10,20,30,40}; req.infer();
    //   EXPECT_EQ(req.get_output_tensor(0).data<int32_t>()[0], 0);
    SUCCEED() << "Skeleton — wire up CPU compile/infer per ov_cpu_unit_tests harness.";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run with ASan enabled: ./ov_cpu_unit_tests --gtest_filter='GatherCpu1DCase.NegativeIndexNoOobRead'. Expected pre-fix failure: AddressSanitizer: heap-buffer-overflow READ of size 4 in ov::intel_cpu::node::Gather::exec1DCase (gather.cpp:987). Post-fix: test passes, out-of-range lane == 0, no ASan report.

## Suggested fix
Add the same upper-bound guard that execReference uses. After the ii-adjustment block (lines 980-985), add: `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` before line 987. This mirrors the execReference pattern at lines 947/959-961 and zeroes out-of-range outputs instead of reading past the buffer.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #140.
