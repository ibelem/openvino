# Security finding #18: At line 987, `pdst[i] = psrc[ii]` is executed after only a negative…

**Summary:** At line 987, `pdst[i] = psrc[ii]` is executed after only a negative…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read of up to (ii - axisDim) * 4 bytes beyond the source data buffer. With axisDim=1 and ii=0x7FFFFFFF this could read gigabytes of process memory, leaking heap/stack content (info leak) or, depending on address layout, triggering a segfault (DoS). Affects any inference call that routes through the 1D fast path.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987` — `Gather::exec1DCase()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** User-supplied indices tensor at GATHER_INDICES port (runtime inference input)

## Description / Root cause
At line 987, `pdst[i] = psrc[ii]` is executed after only a negative-index normalization (lines 980-986). For positive user-supplied indices, there is zero bounds check: `ii >= axisDim` is never tested. `psrc` points to at most 64 i32 elements (the entry gate at line 399 only caps the shape), but `ii` is a raw int32_t value from the user tensor with no upper-bound validation anywhere on the exec1DCase fast path.

**Validator analysis:** The finding is accurate. exec1DCase (gather.cpp:967-989) reads psrc[ii] at line 987 with the only guard being negative-index handling at lines 980-986. CWE-125 OOB read is correct: `ii` is a raw int32_t from the user indices tensor, and for any positive ii >= axisDim the read runs past the source buffer (psrc points to <=64 i32 elements). Notably the negative branch is ALSO defective: when reverseIndexing is false it sets `ii = axisDim`, which is itself one element past the end — so even the 'normalized' path is OOB by one. Every other Gather kernel (execCompressed4Bit line 705, execCompressed8Bit line 791, execReference) gates with `idx < static_cast<size_t>(axisDim)` and zero-fills out-of-range, which exactly the 1D fast path omits. Impact (info leak up to (ii-axisDim)*4 bytes, or segfault DoS) is plausible though bounded by address layout; 'gigabytes' is the theoretical max with ii=0x7FFFFFFF. The proposed fix is correct and sufficient: inserting `if (static_cast<size_t>(ii) >= axisDim) { pdst[i] = 0; continue; }` after line 986 matches the zero-fill semantics already used by execCompressed (lines 960-962) and also closes the ii==axisDim negative-branch bug. A tighter variant would place the check after the negative normalization but mark the reverseIndexing==false case to fill zero rather than write axisDim.

## Exploit / Proof of Concept
1. Build or supply an OpenVINO model containing a v7/v8 Gather node whose data input has static shape [N] (N<=64, dtype i32) and indices input has static shape [M] (M<=64). 2. prepareParams() sets canOptimize1DCase=true because both shape dimensions satisfy the <=64 gate (lines 399-400). 3. At inference time, fill the indices tensor with a value >= N, e.g. index value 1000 with N=10. 4. execute() calls exec1DCase(); the loop reaches `psrc[1000]` — 990 elements past the end of a 10-element buffer — causing an OOB read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 OOB read in Gather::exec1DCase
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:987
//   `pdst[i] = psrc[ii];`  -- no upper-bound check on positive user index.
// The 1D fast path is selected in prepareParams (gather.cpp:396-402) whenever
// the data input is rank<=1, i32, dim<=64 and indices rank<=1, dim<=64.
// Feeding an index >= axisDim must NOT read past the source buffer. Pre-fix
// this triggers an ASan heap-buffer-overflow read; post-fix the out-of-range
// element is zero-filled (consistent with execCompressed gather.cpp:960-962).
//
// Harness: ov_cpu_unit_tests (gtest + ASan), file under intel_cpu/tests/unit/nodes/.
//
// SKELETON: exec1DCase() is a private method and requires a fully wired Node
// graph + allocated MemoryPtr inputs/outputs to reach. Building that inline is
// non-trivial; the cleanest reproduction is a small ov::Model subgraph compiled
// on CPU and run with a crafted indices tensor. TODOs mark the missing pieces.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov;

// TODO: confirm the include path / fixture base used by neighbouring tests in
//       intel_cpu/tests/unit/nodes/ (read an existing *_node_test.cpp first).

TEST(GatherExec1DCaseOOB, PositiveIndexOutOfRangeIsZeroFilledNotOOB) {
    // Build: data[N] i32 (N<=64), indices[M] i32 (M<=64)  -> triggers 1D fast path.
    constexpr size_t N = 10;
    auto data = std::make_shared<opset8::Parameter>(element::i32, Shape{N});
    auto indices = std::make_shared<opset8::Parameter>(element::i32, Shape{1});
    auto axis = opset8::Constant::create(element::i32, Shape{}, {0});
    auto gather = std::make_shared<opset8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{gather},
                                         ParameterVector{data, indices});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> dataVals(N);
    for (size_t i = 0; i < N; ++i) dataVals[i] = static_cast<int32_t>(i);
    Tensor dataT(element::i32, Shape{N}, dataVals.data());

    // Out-of-range positive index: 1000 >> axisDim (10).
    int32_t badIdx = 1000;
    Tensor idxT(element::i32, Shape{1}, &badIdx);

    req.set_input_tensor(0, dataT);
    req.set_input_tensor(1, idxT);

    // Pre-fix: exec1DCase reads psrc[1000] -> ASan heap-buffer-overflow READ.
    // Post-fix: out-of-range index is zero-filled; infer completes cleanly.
    ASSERT_NO_THROW(req.infer());
    auto out = req.get_output_tensor(0);
    EXPECT_EQ(out.data<int32_t>()[0], 0);  // matches execCompressed zero-fill semantics

    // TODO: if the operator contract instead mandates throwing on OOB indices,
    //       replace ASSERT_NO_THROW with EXPECT_THROW(req.infer(), ov::Exception)
    //       and confirm against the chosen fix variant.
    // TODO: also add a negative-index, reverseIndexing==false case to cover the
    //       ii==axisDim off-by-one in gather.cpp:984.
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=GatherExec1DCaseOOB.PositiveIndexOutOfRangeIsZeroFilledNotOOB . Expected pre-fix: ASan 'heap-buffer-overflow READ of size 4' inside Gather::exec1DCase (gather.cpp:987). Expected post-fix: test passes, output element == 0. NOTE: skeleton — verify fixture/include conventions against an existing intel_cpu/tests/unit/nodes test before building; a crafted subgraph (not a private-method call) is required to reach the path.

## Suggested fix
Add an upper-bound clamp/check immediately after the negative-index normalization block, before line 987. For example: after the `if (ii < 0)` block (line 986), add `if (static_cast<size_t>(ii) >= axisDim) { /* handle out-of-range: zero output or skip */ pdst[i] = 0; continue; }`. Alternatively, mirror the bound check used by execReference(): clamp `ii` into [0, axisDim-1] or output a zero/fill value for out-of-range indices, consistent with the operator specification.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #18.
