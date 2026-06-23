# Security finding #53: Line 987 `pdst[i] = psrc[ii]` performs no bounds check on `ii` befo…

**Summary:** Line 987 `pdst[i] = psrc[ii]` performs no bounds check on `ii` befo…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds heap read up to ~8 GB past the source tensor buffer. Attacker can read adjacent heap memory (information disclosure), or trigger a crash (DoS) if the OOB address faults. Because `psrc` is a `uint32_t*` and the read result is written to `pdst`, the leaked memory value is also written to the output tensor and returned to the caller, enabling an info-leak from the inference engine's heap. Affects any model that routes through the 1D optimized path (`dataSrcRank≤1`, 1D data tensor ≤64 elements, 1D index tensor ≤64 elements, data precision i32).
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted int32 index tensor supplied as GATHER_INDICES model input → CPU kernel psrc[] read

## Description / Root cause
Line 987 `pdst[i] = psrc[ii]` performs no bounds check on `ii` before indexing into `psrc`. Two exploitable sub-cases exist: (a) when `reverseIndexing==false` and `pidx[i]<0`, line 984 sets `ii = axisDim` (a `size_t` cast to `int32_t`), making `psrc[axisDim]` read exactly one element past the end of the source buffer; (b) when `pidx[i] >= 0` but `pidx[i] >= axisDim`, `psrc[pidx[i]]` reads arbitrarily far past the buffer (up to INT32_MAX elements). Neither case is guarded. Compare `execReference()` line 947: `if (idx < static_cast<size_t>(axisDim))` — the same pattern used in every other exec path — is absent here.

**Validator analysis:** CONFIRMED. exec1DCase() (gather.cpp:967-989) is the optimized 1D path selected in prepareParams() when dataSrcRank<=1, data precision i32, and both data/index 1D dims <=64 (lines 396-401), then dispatched from execute() at lines 466-468. The loop at 978-988 reads `pidx[i]` (an untrusted int32 from GATHER_INDICES), normalizes only negatives, then does `pdst[i] = psrc[ii]` at 987 with NO range check. (a) reverseIndexing==false + negative idx sets ii=axisDim (line 984) → reads exactly one element past psrc; (b) any pidx[i] in [axisDim, INT32_MAX] reads far past the buffer. The canonical guard `if (idx < static_cast<size_t>(axisDim))` exists in execReference (line 947) but is absent here — a genuine inconsistency, so CWE-125 Out-of-bounds Read is the correct classification. The impact is accurate: psrc is uint32_t* so the leaked heap word is written verbatim into the output tensor and returned to the caller (info-leak), or faults (DoS); max stride is ~INT32_MAX*4 ≈ 8 GB. There is no surrounding try/catch or upstream clamp on index values (only the dim<=64 size gate, which does not bound the index VALUES). The proposed fix is correct and sufficient: it mirrors execReference's behavior, and crucially re-checks `ii >= 0` AFTER the reverseIndexing `ii += axisDim` adjustment (line 982), which also catches the residual-negative case (ii < -axisDim) that the existing code mis-handles. Recommend additionally using `int64_t`/`size_t` comparison to avoid any int32 sign edge, but the proposed guard already covers all reachable cases. Verdict: validated for openvino (defect site) and validated for openvinoEp (model-loading boundary that drives the CPU plugin).

## Exploit / Proof of Concept
Craft an ONNX model with a Gather node whose data input is a 1D int32 tensor of e.g. 4 elements (axisDim=4) and whose indices input is a 1D int32 tensor containing the value 1000000. During inference, `prepareParams()` sets `canOptimize1DCase=true` (dataSrcRank=1, sizes ≤64) and `exec1DCase()` is called. At line 979 `ii=1000000`, the `if (ii<0)` branch is not taken, and at line 987 `psrc[1000000]` reads 4 MB past the 16-byte source buffer, writing heap memory into the output tensor. For the negative-index case: supply index value -1 with `reverseIndexing=false`; line 984 sets `ii=axisDim=4`, and line 987 reads `psrc[4]` — one element (4 bytes) past the end.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987  (Gather::exec1DCase)
// Unchecked line: `pdst[i] = psrc[ii];` with no `ii < axisDim` guard
// (contrast Gather::execReference gather.cpp:947 which clamps).
//
// What this encodes: drive the intel_cpu Gather 1D-optimized path
// (dataSrcRank<=1, i32 data, 1D data & index dims <=64) with an index whose
// value is >= axisDim (and a negative index with reverseIndexing=false).
// PRE-FIX: ASan reports a heap-buffer-overflow READ in Gather::exec1DCase,
// and/or the output element equals leaked adjacent heap memory.
// POST-FIX: the out-of-range lanes are zeroed, so the output is deterministic
// (== 0) and ASan is clean.
//
// HARNESS: ov_cpu_unit_tests (gtest + ASan). Place under
//   openvino/src/plugins/intel_cpu/tests/unit/  next to the existing node
//   single-layer unit tests.
//
// NOTE: emitted as a SKELETON — the exact intel_cpu single-node unit-test
// fixture symbols were not confirmed by reading the test tree, and the 1D
// path requires constructing a real Gather node with a graph/edge context.

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// TODO: confirm the canonical intel_cpu node-level unit-test fixture name and
//       includes (e.g. the helpers used by gather node tests under
//       intel_cpu/tests/unit/). This functional-style variant uses ov::Core +
//       CPU device to force the i32 1D optimized path.
TEST(intel_cpu_gather_exec1DCase, oob_index_must_be_clamped_not_read) {
    // axisDim = 4, 1D i32 data -> hits canOptimize1DCase (prepareParams:396-401)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // Out-of-range index value 1000000 (>= axisDim) AND -1 (negative).
    auto indices = op::v0::Constant::create(element::i32, Shape{2}, {1000000, -1});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    // reverseIndexing defaults false for v8 Gather batch_dims=0.
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, 0);
    auto model   = std::make_shared<Model>(OutputVector{gather->output(0)},
                                           ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{4});
    auto* p = in.data<int32_t>();
    p[0] = 10; p[1] = 11; p[2] = 12; p[3] = 13;
    req.set_input_tensor(in);

    // PRE-FIX: ASan traps inside Gather::exec1DCase reading psrc[1000000]/psrc[4].
    // POST-FIX: out-of-range lanes are zeroed (matches execReference).
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor();
    const auto* o = out.data<int32_t>();
    EXPECT_EQ(o[0], 0);  // index 1000000 -> out of range -> 0
    EXPECT_EQ(o[1], 0);  // index -1, reverseIndexing=false -> axisDim -> 0
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter='intel_cpu_gather_exec1DCase.oob_index_must_be_clamped_not_read'. Expected PRE-FIX: ASan 'heap-buffer-overflow READ of size 4' in ov::intel_cpu::node::Gather::exec1DCase (gather.cpp:987). Expected POST-FIX: clean run, out[0]==out[1]==0. TODO: if the functional ov::Core path is rejected for unit scope, port to the intel_cpu node-level fixture and assert ASSERT_ANY_THROW / zeroed output directly.

## Suggested fix
Add the same bounds guard used in `execReference()` (line 947) inside the loop of `exec1DCase()`. Replace line 987 with:
```cpp
if (ii >= 0 && static_cast<size_t>(ii) < axisDim) {
    pdst[i] = psrc[ii];
} else {
    pdst[i] = 0u;
}
```
This ensures that both negative out-of-range indices (after the `reverseIndexing` adjustment) and positive out-of-range indices are handled safely by zeroing the output rather than reading out-of-bounds, consistent with the behavior of all other exec paths.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #53.
