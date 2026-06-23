# Security finding #145: When `reverseIndexing == true` and the model-supplied index `ii < 0…

**Summary:** When `reverseIndexing == true` and the model-supplied index `ii < 0…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds heap read in the backwards direction: up to `(axisDim + |ii|) * 4` bytes before the data tensor allocation are read and placed in the output tensor. Since `axisDim ≤ 64` and `ii` can be as negative as INT32_MIN, the backward read distance is bounded in practice only by process memory layout. Adjacent heap metadata, other tensor buffers, or freed-block headers could be leaked.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:981` — `Gather::exec1DCase()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX Gather indices tensor flowing from the OpenVINO EP plugin layer into the CPU-node execution path

## Description / Root cause
When `reverseIndexing == true` and the model-supplied index `ii < 0`, the code does `ii += axisDim` (line 982). If `|ii| > axisDim` (e.g., ii = -200 with axisDim ≤ 64), `ii` remains negative after the addition. The code then falls through unconditionally to `psrc[ii]` at line 987. A negative `int32_t` subscript on `const uint32_t* psrc` performs backwards pointer arithmetic, reading memory at address `psrc - |ii|*4`, which is before the start of the source data allocation.

**Validator analysis:** CWE-125 Out-of-bounds Read is accurate. In exec1DCase() the int32_t index `ii` is sign-extended/used directly as a subscript on `const uint32_t* psrc` after `ii += axisDim` (axisDim is a small size_t, <=64, enforced by the canOptimize1DCase gate at gather.cpp:399-401). When the model supplies ii with |ii| > axisDim, ii stays negative and psrc[ii] reads backwards before the buffer; a positive ii > axisDim reads forwards past it. The slow path execReference() (929-963) and the JIT path both clamp/zero-fill out-of-range indices, but the 1D fast path omits that guard entirely — this is a genuine inconsistency, not a mitigated case. Reachability is confirmed: prepareParams() sets canOptimize1DCase for i32 rank<=1 tensors with dims<=64, and execute()/executeDynamicImpl() (466,534) dispatch to exec1DCase(); reverseIndexing defaults to true for v8 Gather (115). Indices are attacker-controlled model data, so the trust boundary is crossed. The proposed fix is correct and sufficient: inserting `if (ii < 0 || static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` after line 982 (better: place it right before line 987 so it also catches positive overshoot) matches execReference's zero-fill semantics and closes both the negative and positive OOB cases. One refinement: the check must run for the non-reverseIndexing branch too (where ii is set to axisDim), and for ii originally >= axisDim — putting the single guard immediately before the psrc[ii] dereference covers all of these.

## Exploit / Proof of Concept
Craft an ONNX model with a 1-D int32 Gather data tensor of shape [1] and a 1-D int32 indices tensor [1] containing value -200 (absolute value exceeds axisDim=1). Set `reverseIndexing=true` (Gather opset with negative-index semantics). During `exec1DCase()`: `ii = -200`, the `ii < 0` branch executes `ii += 1` → `ii = -199`, still negative, no further check, `psrc[-199]` reads 796 bytes before the buffer start.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:982-987 (Gather::exec1DCase)
// Pre-fix: a 1-D i32 Gather (data shape [1], indices [-200], reverseIndexing=true)
//          drives canOptimize1DCase==true (prepareParams:399-401) and execute()
//          dispatches to exec1DCase(), where `ii += axisDim` leaves ii negative and
//          `psrc[ii]` reads before the source allocation -> ASan heap-buffer-overflow.
// Post-fix: the added bounds guard zero-fills the out-of-range element; inference
//           succeeds and the output element equals 0 (matches execReference semantics).
//
// HARNESS: ov_cpu_unit_tests (the intel_cpu component's own unit target;
//          inferred from src/plugins/intel_cpu/tests/unit/CMakeLists.txt:7).
// NOTE: exec1DCase() is private and only reachable by building/executing a Gather
//       subgraph, so this is a best-effort SKELETON driven through the public
//       ov::Core / ov::InferRequest API. // TODO items mark the pieces that must be
//       confirmed against the real headers before this will compile.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

// TODO: confirm the correct fixture/base used by intel_cpu unit tests for
//       end-to-end CPU inference (read tests/unit/ for an existing example that
//       calls core.compile_model(model, "CPU")).
TEST(GatherExec1DCase, NegativeIndexBeyondAxisDimDoesNotReadOOB) {
    using namespace ov;

    // 1-D i32 data of shape [1] -> axisDim = 1 (<=64 enables canOptimize1DCase).
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    // indices = [-200]; |index| > axisDim so ii stays negative after ii += axisDim.
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-200});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    // v8 Gather -> reverseIndexing defaults to true (gather.cpp:115).
    auto gather  = std::make_shared<op::v8::Gather>(data, indices, axis, /*batch_dims*/0);
    auto model   = std::make_shared<Model>(OutputVector{gather->output(0)},
                                           ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor in(element::i32, Shape{1});
    in.data<int32_t>()[0] = 42;
    req.set_input_tensor(in);

    // Pre-fix: ASan aborts here with heap-buffer-overflow (READ) inside exec1DCase.
    ASSERT_NO_THROW(req.infer());

    // Post-fix: out-of-range index is zero-filled (matches execReference 959-963).
    auto out = req.get_output_tensor();
    EXPECT_EQ(out.data<int32_t>()[0], 0);
}
```
**Build / run:** Build: cmake --build <build> --target ov_cpu_unit_tests (build OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./bin/ov_cpu_unit_tests --gtest_filter='GatherExec1DCase.NegativeIndexBeyondAxisDimDoesNotReadOOB'. Expected pre-fix: ASan 'heap-buffer-overflow READ of size 4' in ov::intel_cpu::node::Gather::exec1DCase (gather.cpp:987). Expected post-fix: test passes (output element == 0). TODO: confirm intel_cpu unit-test fixture conventions for compile_model('CPU') before relying on this.

## Suggested fix
After line 982 (`ii += axisDim`), add a guard: `if (ii < 0 || static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }`. This handles both the still-negative case (|original index| > axisDim) and any positive overshoot simultaneously, and is consistent with the zero-fill behavior in `execReference()` (lines 959-963).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #145.
