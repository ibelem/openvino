# Security finding #461: Line 194 performs `m_shape{std::begin(tensor.dims()), std::end(tens…

**Summary:** Line 194 performs `m_shape{std::begin(tensor.dims()), std::end(tens…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error / CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Parsing a crafted ONNX model triggers memory corruption in downstream consumers of the resulting `ov::op::v0::Constant` node whose shape is {SIZE_MAX, SIZE_MAX}. Any code that iterates over shape dimensions, allocates a buffer sized by the shape, or indexes into the tensor will encounter out-of-bounds reads or writes, heap corruption, or a crash (DoS) in the inference application. Because the corrupted constant is inserted into the model graph cache (graph.cpp:132), it can propagate to every downstream graph pass that materializes data from it.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194` — `Tensor::Tensor(const TensorProto&, const std::filesystem::path&, detail::MappedMemoryHandles)()`
**Validated for repos:** openvino
**Trust boundary:** ONNX proto TensorProto.dims (int64_t, attacker-controlled) → ov::Shape (std::vector<size_t>)

## Description / Root cause
Line 194 performs `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` with no negativity or magnitude guard. Each `int64_t` dim value is implicitly narrowed to `size_t`. A negative dim such as -1 becomes SIZE_MAX (18446744073709551615 on 64-bit). No range check is applied before or after the conversion. Downstream at `tensor.cpp:438`, `shape_size(m_shape)` multiplies these size_t values together. When two or more dims wrap (e.g., dims=[-1,-1]), `SIZE_MAX * SIZE_MAX mod 2^64 = 1`, so `shape_elements` becomes 1. The sanity check at `tensor.cpp:467` (`element_count != shape_elements`) then compares `1 != 1`, which is false, silently passing. `ov::op::v0::Constant` is then constructed at line 494 or 480 with a shape containing {SIZE_MAX, SIZE_MAX} and only 1 element of real data — a fully corrupted shape passes the trust boundary into the graph.

**Validator analysis:** The narrowing-conversion bug is real and confirmed by reading the cited code. tensor.hpp:194 builds ov::Shape (vector<size_t>) directly from the signed int64 TensorProto.dims with no `d>=0` check, so -1 becomes SIZE_MAX. The guard at tensor.cpp:467 is genuinely bypassable for the [-1,-1] case because shape_size() does plain modular size_t multiplication (SIZE_MAX*SIZE_MAX mod 2^64 == 1), matching element_count==1, after which ov::op::v0::Constant is constructed at line 494 with a logically corrupt {SIZE_MAX,SIZE_MAX} shape backed by 1 element. CWE-195/CWE-190 categorization is accurate. Note the immediate effect is a logically inconsistent Constant (its internal buffer is shape_size==1 element, so in-bounds element access is fine); the 'memory corruption' impact is realized only in downstream consumers that allocate/iterate per individual dimension rather than via shape_size — the impact statement somewhat overstates certainty of OOB but DoS/invalid-graph from the corrupt shape is credible. The proposed fix (reject any negative dim in the Tensor constructor) is correct, minimal, and sufficient — it stops the corrupt shape at the trust boundary; the secondary shape_size>0 check is a defensible defense-in-depth but the non-negativity check alone is the real root-cause fix. Reachability from the EP boundary is not proven because ORT validates initializer shapes before the EP path, so openvinoEp is rejected; the openvino ONNX frontend directly parses untrusted models and is validated.

## Exploit / Proof of Concept
Craft an ONNX model binary whose initializer TensorProto has `dims: [-1, -1]` and `data_type: DOUBLE` with exactly 8 bytes of `raw_data` (one double element). The Tensor constructor at tensor.hpp:194 stores m_shape={SIZE_MAX, SIZE_MAX}. In `get_ov_constant()` (tensor.cpp:437), `element_count = get_data_size() = 8/8 = 1`. At line 438, `shape_elements = shape_size({SIZE_MAX, SIZE_MAX}) = SIZE_MAX*SIZE_MAX mod 2^64 = 1`. At line 467, `element_count(1) != shape_elements(1)` is false — the check is bypassed. Execution reaches line 494: `std::make_shared<ov::op::v0::Constant>(ov_type, {SIZE_MAX,SIZE_MAX}, get_data_ptr())`, producing a constant with a 2^128-element logical shape backed by 8 bytes of real data. This object is then cached and fed to the rest of the graph (graph.cpp:132), where any operation reading its shape or accessing elements beyond index 0 produces OOB reads or writes.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195/CWE-190 at tensor.hpp:194 (negative ONNX dim narrowed to SIZE_MAX,
// bypassing the size check at tensor.cpp:467 because SIZE_MAX*SIZE_MAX mod 2^64 == 1).
// Pre-fix: convert_model() on a model whose initializer has dims=[-1,-1], data_type=DOUBLE,
//   and 8 bytes of raw_data builds an ov::op::v0::Constant with shape {SIZE_MAX,SIZE_MAX}
//   (no throw) -> corrupt graph. Post-fix: the non-negative-dim check rejects it with ov::Exception.
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
//
// TODO: provide the crafted fixture models/negative_initializer_dims.onnx (or .prototxt converted
//   to .onnx) with: initializer { dims: -1, dims: -1, data_type: DOUBLE, raw_data: <8 bytes> }.
//   A negative value cannot be expressed in a text .prototxt 'dims' the same way as a normal one;
//   it must be authored as a real proto with a negative int64 dim. This is why a self-contained
//   compile-only test is not achievable here.

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_initializer_dims_rejected) {
    // TODO: confirm CommonTestUtils / util::convert_model helper name used by onnx_import.in.cpp
    EXPECT_THROW(convert_model("negative_initializer_dims.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_model_negative_initializer_dims_rejected*'. Requires a crafted negative_initializer_dims.onnx fixture (initializer dims=[-1,-1], DOUBLE, 8 raw_data bytes) placed in the frontend test models dir. Pre-fix: no throw (and ASan may flag downstream OOB once the {SIZE_MAX,SIZE_MAX} Constant is materialized); post-fix: throws ov::Exception from the non-negative dim check at tensor.hpp:194.

## Suggested fix
Before or immediately after constructing `m_shape` in the Tensor constructor (tensor.hpp:194), validate every proto dim: `for (auto d : tensor.dims()) { FRONT_END_GENERAL_CHECK(d >= 0, "Tensor dim must be non-negative, got ", d); }`. Additionally, after computing `shape_elements` at tensor.cpp:438, guard against overflow: `FRONT_END_GENERAL_CHECK(m_shape.empty() || shape_elements > 0, "Shape size overflow detected");`. These two checks together reject the malicious shape before it is ever stored.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #461.
