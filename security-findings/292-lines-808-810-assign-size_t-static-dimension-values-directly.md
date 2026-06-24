# Security finding #292: Lines 808-810 assign `size_t` static dimension values directly to `…

**Summary:** Lines 808-810 assign `size_t` static dimension values directly to `…

**CWE IDs:** CWE-197: Numeric Truncation Error / CWE-681: Incorrect Conversion between Numeric Types
**Severity / Impact:** If a crafted model declares spatial dimensions or kernel sizes ≥ 2^31 (valid as `size_t`), the truncation produces negative `krn`/`src`/`dst` values. This corrupts the `calc_dst` and `paddingR` computations, leading to incorrect (possibly very large or negative) padding values passed to oneDNN — potential memory corruption or DoS. Also affects the index `with_group + 2 + j` if `with_group` participates indirectly and dimensions are borderline.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/conv.cpp:808` — `Convolution::addFusedNode()`
**Validated for repos:** openvino
**Trust boundary:** Model-derived shape dimensions (`size_t`) from getStaticDims() are silently narrowed to `int` with no overflow check at the trust boundary (model load → shape propagation → addFusedNode).

## Description / Root cause
Lines 808-810 assign `size_t` static dimension values directly to `int` variables (`int krn`, `int src`, `int dst`) without any bounds check: `int krn = weightDims[with_group + 2 + j]` (size_t → int), `int src = getInputShapeAtPort(0).getStaticDims()[2 + j]` (size_t → int), `int dst = fusingNode->getOutputShapeAtPort(0).getStaticDims()[2 + j]` (size_t → int). Values larger than INT_MAX (2^31-1) silently wrap to negative integers. On line 812 `krn` is then used in mixed `int*size_t` arithmetic that itself promotes back to `size_t`, and the result is re-assigned to `int krn`, compounding the corruption. Similarly on line 813 `src - krn` (both int, but krn may now be very large or negative) feeds into another mixed-arithmetic expression for `calc_dst`.

**Validator analysis:** The cited code is real: at conv.cpp:808-810 model-derived shape dimensions (size_t from getInputShapeAtPort/getOutputShapeAtPort/getStaticDims and weightDims) are silently narrowed to plain `int` with no bounds check, then used in the padding-recomputation arithmetic (812-814). This is a genuine CWE-197/CWE-681 unchecked narrowing at the model→shape-propagation→addFusedNode path; the block runs at graph-compile time when a depthwise conv is fused, before any activation allocation, so a model declaring a static spatial dim (src) >= 2^31 reaches the truncation without needing to actually allocate a 2^31-element tensor. So the defect is real and reachable in OpenVINO core. HOWEVER the stated impact 'memory corruption' is overstated: the corrupted value only flows into m_attrs.paddingR which is later handed to oneDNN primitive-descriptor creation, which validates conv geometry and would throw/produce a wrong result — the realistic impact is incorrect computation / DoS (exception), not out-of-bounds memory corruption. The proposed fix is correct in direction; the cleaner option the finding itself offers — keeping krn/src/dst/calc_dst as ptrdiff_t throughout the loop — is the best fix because it removes both the truncation and the int*size_t unsigned-promotion hazard at line 814 in one change; an explicit OPENVINO_ASSERT(dim <= INT_MAX) is also acceptable. For the EP repo this is na: the truncation is purely an intel_cpu internal computation, the EP merely passes the graph through.

## Exploit / Proof of Concept
Provide an ONNX model where a fused depthwise convolution node's weight tensor has a spatial dimension set to e.g. 2^31 + 5 (valid as size_t). At line 808: `weightDims[...] = 2147483653` (size_t) → assigned to `int krn` → wraps to `5`. The effective kernel size is now `5` instead of `2^31+5`, producing a wildly incorrect `calc_dst` and subsequently an incorrect `paddingR` passed to oneDNN.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the silent size_t->int narrowing at
// openvino/src/plugins/intel_cpu/src/nodes/conv.cpp:808-810 (Convolution::addFusedNode).
// Pre-fix: a fused depthwise-conv whose input/weight spatial static dim >= 2^31
// truncates `int src/krn/dst` (e.g. 2^31+5 -> 5), yielding a corrupted paddingR;
// post-fix (range-checked conversion / ptrdiff_t): the oversized dim is rejected
// with an ov::Exception instead of being silently wrapped.
//
// SKELETON — exact harness target + graph-builder helpers must be confirmed by
// reading the intel_cpu unit test tree (e.g. src/plugins/intel_cpu/tests/unit/).
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"

// TODO: include the correct intel_cpu unit-test fixture headers used to build a
//       Convolution node + a fusable depthwise Convolution and invoke addFusedNode.
//       The exact target is ov_cpu_unit_tests; symbol names below are placeholders.
TEST(ConvAddFusedNode, RejectsSpatialDimAboveIntMax) {
    // TODO: construct a Convolution node and a depthwise Convolution fusingNode
    //       whose input spatial static dim is (size_t)INT_MAX + 5.
    //       const size_t huge = static_cast<size_t>(std::numeric_limits<int>::max()) + 5;
    //
    // Pre-fix this silently wraps (huge -> 5) inside addFusedNode with no throw.
    // Post-fix the >INT_MAX dim must be rejected.
    // EXPECT_THROW(convNode->addFusedNode(dwConvFusingNode), ov::Exception);
    GTEST_SKIP() << "TODO: wire up intel_cpu Convolution fusion fixture";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run: ov_cpu_unit_tests --gtest_filter=ConvAddFusedNode.RejectsSpatialDimAboveIntMax . Expected pre-fix: no throw (silent truncation of size_t spatial dim to int at conv.cpp:808-810, test fails); post-fix: ov::Exception thrown for dim > INT_MAX so EXPECT_THROW passes. (Skeleton: confirm fixture/builder symbols against src/plugins/intel_cpu/tests/unit/ before compiling.)

## Suggested fix
Replace the plain narrowing assignments with explicit range-checked conversions, e.g.: `OPENVINO_ASSERT(weightDims[with_group + 2 + j] <= INT_MAX, "kernel dim overflow"); int krn = static_cast<int>(weightDims[with_group + 2 + j]);` — and similarly for `src` and `dst`. Alternatively, keep all variables as `ptrdiff_t` throughout the loop body to avoid both the truncation and the unsigned-promotion issue at line 814 in one change.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #292.
