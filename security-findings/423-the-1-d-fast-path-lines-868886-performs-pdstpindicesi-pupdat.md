# Security finding #423: The 1-D fast path (lines 868–886) performs `pdst[pindices[i]] = pup…

**Summary:** The 1-D fast path (lines 868–886) performs `pdst[pindices[i]] = pup…

**CWE IDs:** CWE-129: Improper Validation of Array Index / CWE-787: Out-of-bounds Write
**Severity / Impact:** An attacker who can supply a crafted ONNX/OpenVINO model with a ScatterUpdate node targeting a 1-D data tensor of ≤64 int32 elements can trigger an arbitrary relative out-of-bounds write on the CPU inference thread's heap. A negative index such as -1000000 causes `pdst + (-1000000)` (signed pointer arithmetic), writing far before the destination buffer. A large positive index writes far past it. Either path corrupts adjacent heap metadata or inference tensor data, enabling remote code execution or reliable crash (DoS) in any application using the OpenVINO CPU EP to run user-supplied models.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/scatter_update.cpp:882` — `ScatterUpdate::execute()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model-controlled INDICES_ID memory tensor supplied at inference time

## Description / Root cause
The 1-D fast path (lines 868–886) performs `pdst[pindices[i]] = pupdate[i]` at line 883 without any bounds check on `pindices[i]`. `pindices` is cast directly from the attacker-controlled INDICES tensor buffer (line 880) as `int32_t*`. No check ensures `pindices[i] >= 0` or `pindices[i] < srcLength`. In contrast, the non-optimized path at lines 913–916 uses `CPU_NODE_ASSERT` to enforce `0 <= idxValue < srcDimAxis`. The only pre-execute validation (`initSupportedPrimitiveDescriptors`, lines 199–227) checks tensor *rank* and *shape consistency*, never the *values* stored in the INDICES tensor.

**Validator analysis:** Confirmed: the 1-D fast path (scatter_update.cpp:868-886) is gated only on ranks/dtypes/srcDataDim[0]<=64 (lines 868-869), never on the *values* of pindices. Line 883 does `pdst[pindices[i]] = pupdate[i]` where pindices is a signed int32* over the model-controlled INDICES tensor (line 880). A negative or >=srcLength index yields a relative OOB write on the heap — CWE-129/CWE-787 is accurate, and impact (heap corruption → DoS/possible RCE) is plausible for any model run through the CPU plugin. The contrast with the guarded non-fast path (CPU_NODE_ASSERT 0<=idx<srcDimAxis at 913-915) confirms the missing check is a real omission, not an upstream mitigation; initSupportedPrimitiveDescriptors only validates rank/shape, not values. The proposed fix is correct and sufficient in spirit: insert `CPU_NODE_ASSERT(pindices[i] >= 0 && static_cast<size_t>(pindices[i]) < srcLength, ...)` inside the loop before line 883, mirroring lines 913-916 (negatives are disallowed for ScatterUpdate). The secondary `updateCnt <= srcLength` assert is not strictly required for the OOB (pupdate reads are bounded by updateCnt = update tensor dim), but bounding pdst writes by srcLength is the essential check.

## Exploit / Proof of Concept
Craft a model with: DATA=[0], shape=[1], dtype=i32; INDICES=[-0x10000], shape=[1], dtype=i32; UPDATES=[0xdeadbeef], shape=[1], dtype=i32. At execute() the guard at line 868 is satisfied (srcDataDim.size()==1, indicesDim.size()==1, both i32, srcDataDim[0]==1 ≤ 64). Line 877–879 copies src→dst (safe, bounded by srcLength=1). Line 883 then executes `pdst[-65536] = 0xdeadbeef`, writing 4 bytes 256 KB before the dst buffer on the heap, corrupting allocator metadata or adjacent objects. No path between the trust boundary and this sink validates index values.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:883 (1-D fast-path ScatterUpdate OOB write).
// Pre-fix: an INDICES value outside [0, srcLength) drives pdst[pindices[i]] out of
// bounds -> ASan heap-buffer-overflow. Post-fix: the node must reject the model
// (CPU_NODE_ASSERT/THROW) so the op build/infer throws ov::Exception.
//
// HARNESS: intel_cpu unit tests (target ov_cpu_unit_tests). Place under
//   openvino/src/plugins/intel_cpu/tests/unit/  next to existing single-layer/node tests.
// TODO: confirm exact fixture/helper names by reading the surrounding tests/unit tree;
//       the symbols below are illustrative and must be matched to the repo's harness.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// Builds a model whose ScatterUpdate hits the 1-D fast path: i32 data of rank 1,
// size <= 64, i32 indices, with an out-of-range index value.
TEST(scatter_update_cpu, fast_path_oob_index_is_rejected) {
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-65536}); // OOB
    auto updates = op::v0::Constant::create(element::i32, Shape{1}, {0xdeadbeef});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    int32_t in = 0;
    req.set_input_tensor(Tensor(element::i32, Shape{1}, &in));
    // TODO: depending on where the check lands (compile vs infer), the throw may
    // occur at compile_model or at infer(); assert on the inference call here.
    EXPECT_THROW(req.infer(), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=scatter_update_cpu.fast_path_oob_index_is_rejected . Pre-fix expectation: ASan 'heap-buffer-overflow WRITE of size 4' at scatter_update.cpp:883 (pdst[pindices[i]]). Post-fix expectation: clean run, ov::Exception thrown by the added CPU_NODE_ASSERT and the EXPECT_THROW passes.

## Suggested fix
Add a guard immediately before the scatter loop (after line 881): `if (pindices[i] < 0 || static_cast<size_t>(pindices[i]) >= srcLength) CPU_NODE_THROW("indices value out of range in 1D fast path");` — mirroring the pattern at lines 913–916. Alternatively, reuse `getIndicesValue()` / the existing validation block rather than maintaining a separate unguarded code path. Also add `CPU_NODE_ASSERT(updateCnt <= srcLength, ...)` after line 873 to prevent a secondary scenario where excess updates drive the index loop beyond validation.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #423.
