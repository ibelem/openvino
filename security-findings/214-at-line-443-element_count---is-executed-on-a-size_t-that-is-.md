# Security finding #214: At line 443, `element_count--` is executed on a `size_t` that is 0 …

**Summary:** At line 443, `element_count--` is executed on a `size_t` that is 0 …

**CWE IDs:** CWE-191: Integer Underflow
**Severity / Impact:** Crafted ONNX model causes `ov::op::v0::Constant` to be constructed (line 480 for external-data path, or lines 497/506 for inline path) with `m_shape = {SIZE_MAX}` and a 0-byte data buffer, triggering a SIZE_MAX-element tensor allocation (CWE-789: excessive memory allocation → process crash / DoS) or a massive out-of-bounds read from the empty data pointer (CWE-125). Affects any application that loads an untrusted ONNX model through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:443` — `Tensor::get_ov_constant()()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file → onnx::Tensor::get_ov_constant(): attacker controls tensor data_type (INT4/UINT4), dims array, and absence of raw/external data

## Description / Root cause
At line 443, `element_count--` is executed on a `size_t` that is 0 (produced by `get_data_size()` returning 0 for a nibble-type tensor with no raw data, multiplied by 2 still = 0, then decremented). The unsigned wrap produces SIZE_MAX. Two downstream guards that should catch this are both silently bypassed: (1) the external-data recovery guard at line 462 (`if (element_count == 0 && constant_buffer)`) requires `element_count==0` which SIZE_MAX is not; (2) the shape-mismatch throw at line 467 (`if (element_count != shape_elements)`) is silenced when the attacker simultaneously sets the tensor dims to [-1], which is copied into `ov::Shape` without sign-checking at tensor.hpp:194 (`m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}`), making `shape_elements = shape_size({SIZE_MAX}) = SIZE_MAX` — matching the wrapped `element_count`.

**Validator analysis:** The integer-underflow claim is real and reachable from the OpenVINO ONNX frontend trust boundary (read_model on an untrusted .onnx). get_data_size() returns int32_data_size()=0 for a nibble tensor with no raw/inline data (tensor.hpp:391-414); *2 keeps 0; the odd-shape branch at tensor.cpp:441-443 then decrements an unsigned 0 to SIZE_MAX. The negative dim -1 is copied into ov::Shape (std::vector<size_t>) at tensor.hpp:194 with no sign/bounds check, yielding shape_elements=shape_size({SIZE_MAX})=SIZE_MAX, which is odd (triggers the decrement) and equal to the wrapped element_count (defeats the guard at line 467). The result is std::make_shared<ov::op::v0::Constant> with m_shape={SIZE_MAX} (lines 497/506, inline path NOT wrapped in try/catch — only the external-buffer path at 479-485 is). CWE-191 is accurate. The most realistic impact is CWE-789 excessive allocation → std::bad_alloc DoS (a ~9 EB allocation always fails on 64-bit), rather than a guaranteed CWE-125 OOB read (the allocation fails before any copy), but it is still a genuine defect that defeats an intended size-validation guard. Proposed fix (1) (guard the decrement: throw if element_count==0) is correct and sufficient to stop the underflow on this exact path. Fix (2) (reject negative dims when building m_shape at tensor.hpp:194) is the stronger, root-cause fix and should be preferred, as it also closes other shape-driven overflow paths that consume m_shape; ideally apply both.

## Exploit / Proof of Concept
1. Create an ONNX model with an INT4 or UINT4 initializer tensor. 2. Set `dims: [-1]` (a single int64 dimension of -1; silently cast to SIZE_MAX when stored in `ov::Shape`). 3. Omit `raw_data` and `int32_data` fields entirely (so `get_data_size()` returns `int32_data_size()` = 0). 4. Optionally mark as external data with `data_size=0` to trigger the external-data path. When loaded: `element_count=0*2-1=SIZE_MAX`; recovery guard (line 462) skipped; mismatch check (line 467) silenced because `SIZE_MAX==SIZE_MAX`; Constant constructor called with shape `{SIZE_MAX}` from 0 bytes → allocation of ~8 exabytes or OOB read from empty buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-191 underflow at
//   openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:443 (element_count--)
// and the unchecked negative-dim store at
//   openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194 (m_shape{begin(dims),end(dims)}).
// Pre-fix: a UINT4 initializer with dims=[-1] and no raw/inline data makes
//   element_count underflow to SIZE_MAX == shape_size({SIZE_MAX}), bypassing the
//   size-mismatch throw at tensor.cpp:467, then constructs a Constant with
//   shape {SIZE_MAX} -> std::bad_alloc / OOB (ASan).
// Post-fix: get_ov_constant() (or the Tensor ctor) must reject the input and throw ov::Exception.
//
// HARNESS: ov_onnx_frontend_tests, style of onnx_import.in.cpp (uses convert_model()).
// This requires a crafted binary fixture, so it is a SKELETON.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm helper header that defines convert_model()

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace of convert_model

// TODO: create models/crafted_int4_neg_dim.onnx (read-only fixture):
//   - one initializer, data_type = UINT4 (21) [or INT4 (22)]
//   - dims: [-1]   (single int64 dimension = -1)
//   - NO raw_data, NO int32_data  (so get_data_size()==0)
//   - reference it as a graph output/Identity so get_ov_constant() runs.
TEST(onnx_import_crafted, int4_negative_dim_underflow_is_rejected) {
    // Pre-fix this either underflows element_count to SIZE_MAX and constructs a
    // Constant{shape=SIZE_MAX} (ASan: allocation-size-too-big / SEGV), or silently
    // mis-sizes. Post-fix it must throw a frontend error instead.
    EXPECT_THROW(convert_model("crafted_int4_neg_dim.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (build with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_import_crafted.int4_negative_dim_underflow_is_rejected'. Requires authoring fixture models/crafted_int4_neg_dim.onnx (UINT4 initializer, dims=[-1], no raw/int32 data). Expected pre-fix: ASan 'allocation-size-too-big' or 'requested allocation size exceeds maximum'/SEGV (or std::bad_alloc) when ov::op::v0::Constant is built with shape {SIZE_MAX}; expected post-fix: ov::Exception thrown by get_ov_constant()/Tensor ctor and the EXPECT_THROW passes.

## Suggested fix
Two fixes required: (1) In `get_ov_constant()`, guard the decrement: replace `element_count--;` at line 443 with `if (element_count > 0) element_count--; else { throw error::invalid_external_data("Nibble tensor has zero bytes but odd-element shape for '" + get_name() + "'"); }`. (2) In the Tensor constructor at tensor.hpp:194, validate that all dims are non-negative before storing into `m_shape`: iterate `tensor.dims()` and throw if any value is < 0. Alternatively, change the dims-to-shape conversion to use a checked cast that rejects negative values.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #214.
