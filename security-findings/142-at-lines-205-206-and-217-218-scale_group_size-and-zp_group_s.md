# Security finding #142: At lines 205-206 and 217-218, `scale_group_size` and `zp_group_size…

**Summary:** At lines 205-206 and 217-218, `scale_group_size` and `zp_group_size…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** A crafted model (ONNX or IR) specifying a GatherCompressed node with a SCALE or ZP tensor of shape `{0}` triggers an integer divide-by-zero during graph compilation (before any inference). On x86 this is SIGFPE / process crash — a reliable, unauthenticated denial-of-service against any application that loads untrusted models via OpenVINO's CPU plugin.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205` — `Gather::initSupportedPrimitiveDescriptors()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX/IR model → GatherCompressed node construction → initSupportedPrimitiveDescriptors (called during graph compilation)

## Description / Root cause
At lines 205-206 and 217-218, `scale_group_size` and `zp_group_size` are computed by dividing `getInputShapeAtPort(GATHER_DATA).getElementsCount()` by `getInputShapeAtPort(GATHER_SCALE).getElementsCount()` and `getInputShapeAtPort(GATHER_ZP).getElementsCount()` respectively. `getElementsCount()` (cpu_shape.h:165-173) returns the product of all dimension values, which is 0 for any shape containing a zero-valued dimension (e.g., `{0}` or `{2, 0, 3}`). No non-zero guard precedes either division.

**Validator analysis:** CWE-369 Divide-By-Zero is the accurate classification: gather.cpp:205-206 and 217-218 perform size_t integer division by getInputShapeAtPort(GATHER_SCALE/ZP).getElementsCount(), and cpu_shape.h:165-173 confirms getElementsCount() asserts only that the shape is Static, returning the product of dims (=0 for any zero-valued dimension). On x86 integer div-by-zero raises SIGFPE → process crash, so the DoS impact is accurate, and it occurs at compile time before inference. No mitigation exists: isSupportedOperation returns true for any GatherCompressed (lines 60-61); the constructor only checks data/indices ranks and batchDims/axis; gather_compressed.cpp::validate_and_infer_types runs shape inference only on data/indices/axis (inputs 0-2) and never validates scale (input 3) or zp (input 4) extents. Thus a GatherCompressed with a static {0} scale/zp survives to the division. The proposed fix (a CPU_NODE_ASSERT that scale/zp element count != 0 before each division) is correct and sufficient to convert the crash into a clean ov::Exception; even better is to also reject a zero element count in isSupportedOperation/validate_and_infer_types so the node is rejected earlier. The reachability caveat governs the chain: from the ONNX EP boundary the flaw is NOT reachable (ONNX Gather has no scale/zp; GatherCompressed is an OpenVINO-internal fusion artifact whose scale comes from real per-group quantization constants, never a zero-element tensor), so openvinoEp is rejected and only openvino is validated — reachable from an untrusted OpenVINO IR model directly declaring a GatherCompressed-equivalent subgraph or via internal API.

## Exploit / Proof of Concept
Create a GatherCompressed ONNX node with inputs: DATA of shape {4,8}, INDICES of shape {2}, AXIS constant, and SCALE of shape {0} (zero-element static tensor). When the CPU plugin calls `initSupportedPrimitiveDescriptors` during graph compilation, `getInputShapeAtPort(GATHER_SCALE).getElementsCount()` returns 0 (product of {0} = 0), and the division `DATA_elements / 0` at line 206 raises SIGFPE, crashing the host process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-369 divide-by-zero in
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205-206 and 217-218
// where scale_group_size / zp_group_size are computed by dividing
//   getInputShapeAtPort(GATHER_DATA).getElementsCount()
// by getInputShapeAtPort(GATHER_SCALE/ZP).getElementsCount(), which is 0
// for a static shape containing a zero-valued dimension (cpu_shape.h:165-173).
//
// Pre-fix: constructing a GatherCompressed node with a static {0} decompression
//   scale and compiling it on the CPU plugin triggers SIGFPE inside
//   Gather::initSupportedPrimitiveDescriptors (no clean throw).
// Post-fix: the node is rejected with an ov::Exception (CPU_NODE_ASSERT) instead.
//
// HARNESS: ov_cpu_unit_tests (intel_cpu component gtest target).
// TODO: confirm the exact target name and an existing test fixture by reading
//       targets/openvino/src/plugins/intel_cpu/tests/unit/ (could not enumerate
//       the directory with the read-only tool).

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/gather_compressed.hpp"

using namespace ov;

// TODO: verify GatherCompressed constructor signature against
//       targets/openvino/src/common/transformations/include/ov_ops/gather_compressed.hpp
TEST(GatherCompressedCpuRegression, ZeroElementScaleIsRejectedNotSigfpe) {
    // DATA {4,8} u8, INDICES {2} i32, AXIS const 0, SCALE {0} f32 (zero elements).
    auto data    = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{4, 8});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    auto axis    = op::v0::Constant::create(element::i64, Shape{}, {0});
    // Zero-element static scale tensor -> getElementsCount() == 0.
    auto scale   = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});

    auto gather_compressed = std::make_shared<op::internal::GatherCompressed>(
        data, indices, axis, /*batch_dims=*/0, scale);

    auto model = std::make_shared<Model>(OutputVector{gather_compressed},
                                         ParameterVector{data, indices},
                                         "gather_compressed_zero_scale");

    ov::Core core;
    // Pre-fix this compilation path divides by zero in
    // Gather::initSupportedPrimitiveDescriptors -> SIGFPE.
    // Post-fix it must throw a catchable ov::Exception.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ./ov_cpu_unit_tests --gtest_filter='GatherCompressedCpuRegression.ZeroElementScaleIsRejectedNotSigfpe'. Pre-fix expected failure: process aborts with SIGFPE (integer divide-by-zero) inside Gather::initSupportedPrimitiveDescriptors at gather.cpp:206 (under a debugger the FPE points at the size_t division). Post-fix expected: the test passes because the node is rejected with ov::Exception via the added CPU_NODE_ASSERT on scale/zp element count. TODO: adjust target/fixture names after reading intel_cpu/tests/unit/.

## Suggested fix
Add a non-zero check before each division. For example:
```cpp
const size_t scale_elem_count = getInputShapeAtPort(GATHER_SCALE).getElementsCount();
CPU_NODE_ASSERT(scale_elem_count != 0, "GATHER_SCALE tensor must have at least one element");
scale_group_size = getInputShapeAtPort(GATHER_DATA).getElementsCount() / scale_elem_count;
```
And similarly for `GATHER_ZP` at lines 217-218. Alternatively, validate that neither SCALE nor ZP has any zero dimension during `isSupportedOperation` or the Gather constructor.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #142.
