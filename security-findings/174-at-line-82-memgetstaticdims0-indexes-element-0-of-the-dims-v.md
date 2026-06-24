# Security finding #174: At line 82, `mem.getStaticDims()[0]` indexes element 0 of the dims …

**Summary:** At line 82, `mem.getStaticDims()[0]` indexes element 0 of the dims …

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** The garbage `size_t` value read from adjacent heap memory is passed directly to `lastSecondInputValues.resize(N, 0)` (line 82). This can (a) trigger an enormous allocation causing process OOM / denial-of-service, or (b) corrupt the heap-allocator metadata, giving an attacker a primitive to influence subsequent allocations. Reachable from any caller that feeds the Intel CPU EP an ONNX model with a static scalar Reshape shape input.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/reshape.cpp:82` — `Reshape::needShapeInfer()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model (rank-0 / scalar shape tensor as second input) supplied via ORT OpenVINO EP → OpenVINO Intel CPU plugin Reshape node

## Description / Root cause
At line 82, `mem.getStaticDims()[0]` indexes element 0 of the dims vector returned for the second-input memory with no prior rank/size check. `getStaticDims()` (cpu_shape.h:127-130) only asserts `type == ShapeType::Static` then returns `minDims` unchanged; for a rank-0 scalar it returns an empty `VectorDims`. `operator[]` on an empty `std::vector` at index 0 is undefined behaviour — a heap OOB read. The constructor guard (lines 56-76) only rejects second inputs whose `PartialShape::is_dynamic()` is true; a static scalar (`is_dynamic()` == false, rank == 0) passes silently. The ORT OpenVINO EP (`ov_provider.cc`) converts the ONNX graph to a proto string and hands it to the OpenVINO ONNX frontend without any per-op validation of the Reshape second-input rank, so the malformed graph reaches the CPU plugin unchanged.

**Validator analysis:** The CWE-125 OOB read is accurate as a defect: getStaticDims() (cpu_shape.h:127-130) returns minDims unchanged, which is empty for a rank-0 scalar, and std::vector::operator[](0) on an empty vector is UB — a heap OOB read of a size_t that is then fed to vector::resize. Reachability holds only when the node is dynamic (needShapeInfer is an override invoked for dynamic nodes), which requires a dynamic data input (input 0); with a static scalar pattern on input 1 the constructor guard at line 59 (is_dynamic()==false) and core shape inference (lines 273-292 explicitly allow rank-0 pattern, requiring value==1) both pass, so the malformed-but-static-scalar graph survives to inference. The 'heap corruption / attacker primitive' part of the impact is overstated — the read value is consumed only as a resize count, so the realistic impact is a read of adjacent stack/heap garbage leading to a huge allocation (OOM/DoS) or an immediate crash; it is not a controllable write primitive. The proposed fix is correct and sufficient at the plugin layer: fix #1 (guard `dims.empty()` before indexing in needShapeInfer) is the minimal and correct patch; fix #2 (reject rank-0 static second input in the constructor) is a good defense-in-depth addition. Fix #3 in the EP is unnecessary for correctness (the EP is not where the bug lives) and better handled by the core/plugin guard; better still, the core could reject rank-0 patterns uniformly rather than special-casing value==1. The vulnType (CWE-125) is correct; impact should be downgraded to DoS/OOB-read rather than heap-corruption primitive.

## Exploit / Proof of Concept
Craft an ONNX model containing `Reshape(data, shape)` where `shape` is a scalar `INT64` initializer (shape = `[]`, rank 0, value e.g. 1). The ORT OpenVINO EP accepts this — `ov_provider.cc::GetCapability` checks only whether the op type is supported, not its input rank. The CPU plugin Reshape constructor's guard (line 59) queries `get_input_partial_shape(1).is_dynamic()` which is false for a static scalar, so no exception is thrown. When `needShapeInfer()` is first called at inference time, `lastSecondInputValues.empty()` is true, execution reaches line 82, `mem.getStaticDims()` returns an empty vector, and `[0]` reads out-of-bounds heap data. That value becomes the resize count passed to `std::vector::resize`, causing DoS or heap corruption.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read at
//   openvino/src/plugins/intel_cpu/src/nodes/reshape.cpp:82
//   `lastSecondInputValues.resize(mem.getStaticDims()[0], 0);`
// where getStaticDims() (cpu_shape.h:127-130) returns an EMPTY VectorDims for a
// rank-0 (scalar) static second input, so operator[](0) reads out of bounds.
//
// This encodes the fix: a Reshape node whose second (shape) input is a static
// rank-0 scalar must be REJECTED (CPU_NODE_THROW) instead of dereferencing
// element [0] of an empty dims vector. Pre-fix this aborts under ASan with a
// container-overflow / heap-buffer-overflow in needShapeInfer(); post-fix the
// empty-dims guard throws ov::Exception during node construction or shape infer.
//
// NOTE (why this is a SKELETON): exercising Node::needShapeInfer() in isolation
// requires a fully built ov::intel_cpu::Graph with allocated parent edges and
// backing Memory for input 1, which the read-only tree does not expose via a
// simple unit entry point. The exact construction helpers must be copied from
// an existing ov_cpu_unit_tests fixture. TODOs below name what is missing.

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

using namespace ov;

TEST(ReshapeNodeOOB, scalar_shape_input_is_rejected_not_oob) {
    // Build a v1::Reshape with a DYNAMIC data input (so the CPU node is dynamic
    // and needShapeInfer() is on the hot path) and a STATIC RANK-0 scalar shape
    // pattern (PartialShape{} -> is_dynamic()==false, rank()==0). This is the
    // exact shape that slips past the constructor guard at reshape.cpp:59.
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto pattern = op::v0::Constant::create(element::i64, ov::Shape{} /* rank-0 scalar */, {1});
    const auto reshape = std::make_shared<op::v1::Reshape>(data, pattern, /*special_zero=*/true);

    // TODO: build the intel_cpu Graph node from `reshape` using the same helper
    //       used in intel_cpu/tests/unit (e.g. cpuNodeFromOp / GraphContext) and
    //       allocate a backing Memory for parent edge 1 with a rank-0 descriptor.
    // TODO: replace the line below with the node construction + needShapeInfer()
    //       invocation once the helper symbol is confirmed by reading the
    //       surrounding intel_cpu unit test fixtures.
    //
    // Expected post-fix behaviour: rank-0 second input is rejected.
    //   EXPECT_THROW(makeAndInferCpuReshape(reshape), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu Graph/Node fixture to call needShapeInfer()";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_cpu_unit_tests --gtest_filter=ReshapeNodeOOB.scalar_shape_input_is_rejected_not_oob . Expected pre-fix: AddressSanitizer 'container-overflow' / 'heap-buffer-overflow READ of size 8' inside ov::intel_cpu::node::Reshape::needShapeInfer (reshape.cpp:82). Expected post-fix: the rank-0 second input is rejected with ov::Exception (CPU_NODE_THROW), no ASan error. NOTE: skeleton — complete the TODO node-construction wiring before the test will compile/run.

## Suggested fix
1. In `needShapeInfer()` (line 82) add a rank guard before indexing: `const auto& dims = mem.getStaticDims(); if (dims.empty()) { CPU_NODE_THROW("second input must be at least 1-D"); } lastSecondInputValues.resize(dims[0], 0);`
2. In the constructor (lines 58-61) also reject rank-0 static second inputs: after checking `is_dynamic()`, add `if (op->get_input_partial_shape(1).rank().is_static() && op->get_input_partial_shape(1).rank().get_length() == 0) { CPU_NODE_THROW("has scalar (rank-0) second input"); }`
3. In the ORT OpenVINO EP (`ov_provider.cc`), before handing a fused graph to `CompileEpContextNode`, validate that any Reshape/Squeeze/Unsqueeze node's shape input is at least 1-D, returning `ORT_EP_FAIL` on violation to prevent the malformed graph from reaching the CPU plugin.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #174.
