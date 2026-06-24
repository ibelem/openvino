# Security finding #225: `shape_size(m_shape)` at line 438 multiplies all `size_t` dimension…

**Summary:** `shape_size(m_shape)` at line 438 multiplies all `size_t` dimension…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** The mismatch guard — the sole gatekeeper preventing a corrupted shape from reaching `ov::op::v0::Constant` — is defeated. The Constant is created with a shape containing SIZE_MAX dimensions and a small real data buffer. Any subsequent operation that resolves the shape for memory allocation or element access will either attempt a massive allocation (DoS) or perform an out-of-bounds memory read/write (potential RCE/info-leak). Affects any application that loads a crafted ONNX model through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:438` — `Tensor::get_ov_constant()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled `m_shape` (populated from untrusted protobuf dims) used in unchecked `size_t` multiplication via `shape_size(m_shape)` at line 438

## Description / Root cause
`shape_size(m_shape)` at line 438 multiplies all `size_t` dimensions using standard unsigned arithmetic with no overflow detection. When `m_shape` contains SIZE_MAX-valued elements (from negative protobuf dims as described above), the product wraps around modulo 2^64. The result can accidentally equal `element_count` (the true data element count), causing the mismatch guard at line 467 (`element_count != shape_elements`) to evaluate to `false` and silently pass, allowing a malformed shape through to `ov::op::v0::Constant` construction.

**Validator analysis:** The CWE-190 categorization is accurate: ov::shape_size performs unchecked size_t multiplication, and the constructor at tensor.hpp:194 admits negative protobuf dims (converted to ~SIZE_MAX) with no validation. For a crafted initializer with dims=[-1,-1] and one data element, (2^64-1)^2 mod 2^64 == 1 == element_count, so the sole mismatch guard at line 467 is defeated and a Constant with a {SIZE_MAX,SIZE_MAX} shape is built. Stated impact (massive allocation DoS / potential OOB on later shape resolution) is plausible; calling it RCE is optimistic since the immediate Constant copy uses the same overflowed (=1) element count and copies only one element. The proposed fix is partially correct but the better/root fix is the sign-and-range validation of each dim at the source (tensor.hpp:194): reject any negative/oversized dim before it is cast to size_t. A checked-multiply shape_size helps but only catches the overflow case, not the equally-broken non-overflowing-but-bogus shapes; the SIZE_MAX/ov_type.size() guard alone can be bypassed by an attacker-tuned element count, so per-dimension bounds validation at construction is the necessary fix. openvinoEp is rejected because the defect lives entirely in openvino and is not reachable from the EP boundary (ORT validates initializer dims first).

## Exploit / Proof of Concept
As above: dims=[-1,-1] with 1 float element gives shape_size=1=element_count; the check `1 != 1` is false, guard skipped. Then line 494 constructs `ov::op::v0::Constant(ov_type, {SIZE_MAX, SIZE_MAX}, data_ptr_to_1_float)`. Any reshape/inference step that performs `shape_size({SIZE_MAX, SIZE_MAX}) * sizeof(float)` = 4 bytes (overflow again), and then iterates over SIZE_MAX elements, reads far beyond the 4-byte buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 in onnx frontend tensor.cpp:438-467 / tensor.hpp:194.
// Pre-fix: a crafted ONNX initializer with negative dims (e.g. dims=[-1,-1]) is
// copied into ov::Shape as {SIZE_MAX,SIZE_MAX}; shape_size() wraps to a value that
// matches element_count, defeating the guard at tensor.cpp:467 and constructing an
// ov::op::v0::Constant with a SIZE_MAX shape over a 1-element buffer.
// Post-fix: per-dimension sign/range validation (tensor.hpp:194) or a checked
// shape_size must reject the model with an ov::Exception.
//
// Harness: ov_onnx_frontend_tests (gtest), in the style of onnx_import.in.cpp.
// NOTE: this requires a crafted binary fixture (negative initializer dims) that
// cannot be authored as plain text here -> emitted as a SKELETON.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: provide models/initializer_negative_dims.onnx containing an initializer
//       whose TensorProto.dims = [-1, -1] with raw_data holding exactly 1 float.
//       Build it with onnx.helper / a hex editor; place under the frontend test
//       models dir consumed by convert_model().
OPENVINO_TEST(${BACKEND_NAME}, onnx_initializer_negative_dims_rejected) {
    // TODO: confirm convert_model() helper signature in onnx_utils.hpp for this tree.
    EXPECT_THROW(convert_model("initializer_negative_dims.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_initializer_negative_dims_rejected*. Requires crafting models/initializer_negative_dims.onnx (initializer dims=[-1,-1], 1 float of data). Pre-fix the test FAILS (no throw; a Constant with a SIZE_MAX shape is built and may trip an ASan/UBSan size_t-overflow or massive-allocation report downstream); post-fix it PASSES because dim validation throws ov::Exception.

## Suggested fix
After computing `shape_elements` at line 438, add an overflow and sanity check: `FRONT_END_GENERAL_CHECK(shape_elements <= SIZE_MAX / ov_type.size(), "shape_size overflows or yields implausibly large element count");`. More robustly, implement `shape_size` using checked multiplication (e.g., loop with `__builtin_mul_overflow` or a saturating multiply that throws on overflow). Additionally, validate that each element of `m_shape` is within a sane bound (e.g., < 2^32) before the product is computed. The root fix remains the sign-check at tensor.hpp:194.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #225.
