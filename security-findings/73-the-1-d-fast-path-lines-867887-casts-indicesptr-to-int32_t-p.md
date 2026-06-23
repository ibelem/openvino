# Security finding #73: The 1-D fast-path (lines 867–887) casts `indicesPtr` to `int32_t* p…

**Summary:** The 1-D fast-path (lines 867–887) casts `indicesPtr` to `int32_t* p…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Attacker-controlled write of an arbitrary int32 value to an arbitrary heap offset (up to INT32_MAX * 4 bytes past, or negative offsets before, the destination buffer). This can corrupt adjacent heap metadata or object pointers, leading to denial-of-service or, on a crafted allocation layout, remote code execution. Affects any application that loads an untrusted OpenVINO/ONNX model containing a ScatterUpdate node with a 1-D int32 data tensor ≤64 elements and a crafted indices tensor.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883` — `ScatterUpdate::execute()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX/IR model-supplied indices tensor (int32, ≤1-D shape) ingested by the ScatterUpdate CPU node

## Description / Root cause
The 1-D fast-path (lines 867–887) casts `indicesPtr` to `int32_t* pindices` and writes `pdst[pindices[i]] = pupdate[i]` at line 883 with no bounds check on `pindices[i]`. The destination buffer `pdst` is at most `srcDataDim[0]` (≤64) elements wide. Any negative or out-of-range index value silently causes a heap write outside that buffer. The `return` at line 885 exits before the `if (axisRelaxed)` block (lines 890–947) that contains the only `CPU_NODE_ASSERT` index-range checks (lines 913–916), so those guards are never reached on this path.

**Validator analysis:** Confirmed by reading lines 852-947. The 1-D fast path (867-887) is entered when mode==ScatterUpdate, src is 1-D i32 ≤64 elems, indices ≤1-D i32, updates ≤1-D. It copies src→dst then performs pdst[pindices[i]]=pupdate[i] (line 883) with NO validation of pindices[i], and unconditionally `return`s at line 885 — bypassing the axisRelaxed block whose CPU_NODE_ASSERT (913-916) is the only index range guard. A negative or huge index yields an attacker-controlled int32 store at an arbitrary heap offset. The vulnType CWE-787 Out-of-bounds Write is accurate, and the impact (heap corruption → DoS / potential RCE) is fair for an IR-model trust boundary. The proposed fix is correct and sufficient: it mirrors the general-path guard; note the general ScatterUpdate path forbids negative indices (idxValue>=0), so checking idx>=0 && (size_t)idx<srcDataDim[0] is the right semantic. One refinement: also guard against updateCnt exceeding the indices/update buffer extents, but the index bounds check is the core fix. I reject openvinoEp because reachability through the ONNX EP cannot be confirmed — ONNX scatter ops do not lower to ScatterUpdate mode, which is the precondition for this fast path; the defect is real but propagates only from an IR-level ScatterUpdate node (openvino).

## Exploit / Proof of Concept
Craft a model with a ScatterUpdate node where: data shape=[4] (dtype i32), indices shape=[1] (dtype i32, value=-1 or value=1000000), updates shape=[1] (dtype i32). The fast-path gate (line 868–869) is satisfied (1-D, i32, srcDataDim[0]=4 ≤ 64). At line 883, `pdst[pindices[0]]` evaluates to `pdst[-1]` (4 bytes before the buffer) or `pdst[1000000]` (4 MB past the buffer), writing the update value there with no error raised.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-787 OOB write in
// openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:883 (ScatterUpdate::execute 1-D fast path).
// Pre-fix: with data=[4] i32, indices=[1] i32 value=1000000 (or -1), the fast path executes
//   pdst[pindices[0]] = pupdate[0] with no bounds check -> ASan heap-buffer-overflow WRITE.
// Post-fix: the added CPU_NODE_ASSERT(idx>=0 && idx<srcDataDim[0]) rejects the model -> ov::Exception.
//
// TODO(harness): confirm exact target name (likely ov_cpu_unit_tests) and the model-building
//   helpers used under openvino/src/plugins/intel_cpu/tests/unit/ . This is a SKELETON: the CPU
//   node is normally driven via a compiled ov::Model + InferRequest, not constructed directly,
//   so the cleanest reproduction is an end-to-end infer on a tiny ScatterUpdate graph.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_scatter_update, oob_index_1d_fastpath_is_rejected) {
    // data: 1-D i32, <=64 elements -> triggers the fast-path gate (scatter_update.cpp:868-869)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // crafted out-of-range index (1000000) on the destination buffer of width 4
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {1000000});
    auto updates = op::v0::Constant::create(element::i32, Shape{1}, {0x41414141});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    Tensor in(element::i32, Shape{4});
    std::fill_n(in.data<int32_t>(), 4, 0);
    req.set_input_tensor(in);

    // Pre-fix: infer() performs an out-of-bounds heap write (ASan aborts here).
    // Post-fix: the bounds check throws before any write.
    EXPECT_THROW(req.infer(), ov::Exception);
    // TODO: if v3::ScatterUpdate constant-folds the index before reaching the CPU node,
    //   replace `indices`/`updates` with Parameters and feed the crafted index at runtime.
}
```
**Build / run:** Build: cmake --build build --target ov_cpu_unit_tests. Run: ./bin/intel64/Release/ov_cpu_unit_tests --gtest_filter='intel_cpu_scatter_update.oob_index_1d_fastpath_is_rejected'. Expected pre-fix (ASan build): 'ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 4' inside ScatterUpdate::execute at scatter_update.cpp:883. Expected post-fix: the test passes because req.infer() throws ov::Exception from the new index-range CPU_NODE_ASSERT.

## Suggested fix
Add an explicit bounds check inside the fast-path loop before the write. For example, replace the loop at lines 882–884 with:
```cpp
for (size_t i = 0; i < updateCnt; i++) {
    int32_t idx = pindices[i];
    CPU_NODE_ASSERT(idx >= 0 && static_cast<size_t>(idx) < srcDataDim[0],
        "indices value out of bounds in 1-D fast path");
    pdst[idx] = pupdate[i];
}
```
This mirrors the check already present in the general path at lines 913–916.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #73.
