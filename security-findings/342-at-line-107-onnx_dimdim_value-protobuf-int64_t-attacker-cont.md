# Security finding #342: At line 107, `onnx_dim.dim_value()` (protobuf `int64_t`, attacker-c…

**Summary:** At line 107, `onnx_dim.dim_value()` (protobuf `int64_t`, attacker-c…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** A crafted ONNX model can supply negative dim_values (other than -1) for any tensor dimension. These are silently accepted and mapped to a static zero-length dimension rather than triggering an error or being treated as dynamic. Downstream inference code that allocates memory based on the resolved shape may allocate zero-byte buffers for those dimensions, then attempt to read/write them, potentially causing heap corruption, out-of-bounds access, or incorrect model execution. An attacker who can supply a malicious ONNX model to any application using this frontend can trigger this path.
**Affected location:** `targets/openvino/src/frontends/onnx/onnx_common/src/utils.cpp:107` — `onnx_to_ov_shape()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model bytes → onnx_to_ov_shape at the ONNX frontend ingestion point

## Description / Root cause
At line 107, `onnx_dim.dim_value()` (protobuf `int64_t`, attacker-controlled) is passed directly to `ov::Dimension(int64_t)` with no non-negative check. The `Dimension` constructor (dimension.cpp:71) only treats the special value `-1` as a dynamic dimension (`Interval(0, s_max)`). For any other negative value (e.g., `-2`, `-99`), it constructs `Interval(negative, negative)`, which `Interval::canonicalize()` (interval.cpp:52-53) then silently clips both bounds to 0 via `clip()`. The result is a static `Dimension(0)` accepted without error instead of being rejected as an invalid ONNX shape.

**Validator analysis:** The data flow is confirmed: onnx_to_ov_shape (utils.cpp:99-114) enters the has_dim_value() branch for any TensorShapeProto dim carrying a dim_value, and emplaces ov::Dimension(int64_t) without a non-negative check. Dimension(value_type) (dimension.cpp:70-71) maps only -1 to the dynamic Interval(0, s_max); any other negative value forms Interval(neg, neg), and Interval::canonicalize() (interval.cpp:47-54) takes the else branch (m_max_val==m_min_val, not <), clipping both bounds to 0 via clip() (interval.cpp:10-12). Result: dim_value=-2 silently becomes static Dimension(0), accepted as a valid shape. CWE-20 (Improper Input Validation) is the correct categorization and the flaw is real and reachable from untrusted ONNX model bytes. However the stated impact is overstated/speculative: the concrete observable effect is acceptance of an invalid ONNX shape collapsed to a zero-length static dimension, which typically yields empty tensors or downstream shape-inference errors, not a demonstrated heap corruption / OOB — no proof of an exploitable memory-safety bug is given. The proposed fix (OPENVINO_ASSERT(dim_val >= 0, ...)) is correct and sufficient: valid ONNX never encodes dynamic dims via dim_value (it uses dim_param), so dim_value should always be >=0 and rejecting negatives outright is appropriate; the -1 OpenVINO convention does not apply to ONNX dim_value and need not be preserved here.

## Exploit / Proof of Concept
Craft an ONNX model where a tensor's `TensorShapeProto` contains a `dim` entry with `dim_value = -2` (or any negative value other than -1). When parsed, `onnx_dim.has_dim_value()` is true, so the `else` (dynamic) branch is not taken; instead `dims.emplace_back(-2)` is called. Inside `Dimension(-2)`: since `-2 != -1`, constructs `Interval(-2, -2)` → `canonicalize()` clips both to 0 → static `Dimension(0)`. The PartialShape is returned silently as valid. Any subsequent call to `.get_shape()` on this shape yields `size_t 0` for that dimension with no error, and buffer allocations downstream use this incorrect size.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for utils.cpp:107 (ov::frontend::onnx::common::onnx_to_ov_shape).
// Encodes the fix: a TensorShapeProto dim carrying a negative static dim_value
// (other than the -1 OpenVINO-dynamic convention, which ONNX never uses for
// dim_value) must be REJECTED, not silently clipped to a static Dimension(0)
// by Interval::canonicalize() (interval.cpp:47-54).
//
// Pre-fix: onnx_to_ov_shape returns PartialShape{0} for dim_value=-2 (no throw)
//          -> EXPECT_THROW fails.
// Post-fix (OPENVINO_ASSERT(dim_val >= 0, ...)): the call throws ov::Exception
//          -> EXPECT_THROW passes.
//
// NOTE: onnx_to_ov_shape takes onnx::TensorShapeProto, which requires linking
// the onnx_common protobuf headers. If those symbols are not exposed to the
// onnx_import test target, fall back to the convert_model("crafted.onnx")
// pattern of onnx_import.in.cpp with a fixture whose value_info encodes
// dim_value: -2. TODOs below mark what must be confirmed against the tree.

#include <gtest/gtest.h>
#include "openvino/core/except.hpp"

// TODO: confirm the public header that declares onnx_to_ov_shape and pulls in
//       the generated onnx.proto TensorShapeProto (e.g. "onnx_common/utils.hpp").
// #include "onnx_common/utils.hpp"

using namespace ov::frontend::onnx::common;

TEST(onnx_common_utils, onnx_to_ov_shape_rejects_negative_dim_value) {
    // TODO: replace with the real generated protobuf type / namespace.
    // ::onnx::TensorShapeProto shape;
    // auto* d = shape.add_dim();
    // d->set_dim_value(-2);  // attacker-controlled negative static dim
    //
    // // Pre-fix this silently yields PartialShape{0}; post-fix it must throw.
    // EXPECT_THROW(onnx_to_ov_shape(shape), ov::Exception);
    GTEST_SKIP() << "TODO: wire up TensorShapeProto include for onnx_common test target";
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (gtest + ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_common_utils.onnx_to_ov_shape_rejects_negative_dim_value . Pre-fix the EXPECT_THROW fails (onnx_to_ov_shape returns a static Dimension(0) without throwing); post-fix (OPENVINO_ASSERT(dim_val >= 0) at utils.cpp:106-107) it throws ov::Exception and the test passes. If the protobuf symbol cannot be linked, switch to convert_model with a crafted .onnx whose value_info dim has dim_value:-2 and wrap in EXPECT_THROW(..., ov::Exception).

## Suggested fix
Add a non-negative guard in the `has_dim_value()` branch before constructing `ov::Dimension`. For example:
```cpp
if (onnx_dim.has_dim_value()) {
    const auto dim_val = onnx_dim.dim_value();
    OPENVINO_ASSERT(dim_val >= 0,
        "ONNX model contains a negative static dimension value: ", dim_val);
    dims.emplace_back(dim_val);
}
```
Alternatively, treat any negative value (other than the ONNX-convention `-1` for dynamic) as a dynamic dimension: `dims.emplace_back(dim_val < 0 ? ov::Dimension::dynamic() : ov::Dimension(dim_val));` — but rejecting outright is preferable because ONNX does not define semantics for negative static dims.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #342.
