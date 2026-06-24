# Security finding #228: Line 121: `return {std::begin(attribute.ints()), std::end(attribute…

**Summary:** Line 121: `return {std::begin(attribute.ints()), std::end(attribute…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error
**Severity / Impact:** Same as Finding 1 but affects the more common vector-attribute path (e.g. `kernel_shape`, `strides`, `dilations`, `pads`). Provides a wider attack surface: any multi-dimensional pool or convolution attribute that is a list of integers can be attacked. The resulting huge `size_t` values in `ov::Strides` / `ov::Shape` lead to integer overflow in output shape arithmetic or out-of-bounds buffer accesses during inference execution.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/attribute.hpp:121` — `detail::attribute::get_value<std::vector<std::size_t>>()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-supplied ONNX model protobuf: AttributeProto::ints() (repeated int64_t) → std::vector<std::size_t> returned to operator constructors

## Description / Root cause
Line 121: `return {std::begin(attribute.ints()), std::end(attribute.ints())}` constructs a `std::vector<std::size_t>` directly from a range of `int64_t` values via an implicit narrowing conversion. Negative INTS values (e.g. -1 in `kernel_shape` or `strides`) silently become SIZE_MAX-range size_t values. No upstream check exists in `node.cpp:188-196`.

**Validator analysis:** The cited defect is real: line 121 `return {std::begin(attribute.ints()), std::end(attribute.ints())}` builds a vector<size_t> directly from int64_t protobuf ints with no range validation, so a negative element (e.g. -1) silently becomes SIZE_MAX. CWE-195 (signed→unsigned conversion) is the accurate classification. It is reachable from an attacker ONNX model: convpool::get_kernel_shape (convpool.cpp:29) and get_strides/get_dilations via get_attribute_value (convpool.cpp:63) call get_attribute_value<std::vector<size_t>>, and there is no upstream non-negative check. The IMPACT is partially overstated: the SIZE_MAX value lands in ov::Shape/ov::Strides, and downstream Conv/MaxPool shape inference uses signed Dimension arithmetic that will in practice throw a NodeValidationFailure rather than reliably corrupt the heap during import — true OOB/heap corruption is speculative, not demonstrated. The defect is still genuine (wrong silent conversion that yields nonsensical huge dims and an unclear failure). The proposed fix is correct and sufficient: validate each element is non-negative before static_cast in the INTS case; it should also be applied symmetrically to the single-INT case (line 119) and ideally to other size_t conversions (get_value<size_t> at line 112). Use a frontend-style throw (CHECK_VALID_NODE/OPENVINO_THROW) so the error is reported as a clean import failure.

## Exploit / Proof of Concept
Set any INTS attribute (e.g. `kernel_shape=[3,-1]`) in an ONNX Conv or Pool node. `convpool::get_kernel_shape` (convpool.cpp:29) calls `node.get_attribute_value<std::vector<size_t>>("kernel_shape", ...)`, which reaches `get_value<std::vector<std::size_t>>` → line 121 range constructor. The -1 element is implicitly converted to SIZE_MAX, producing a vector `{3, 18446744073709551615}`. This dimension value is then used in output-shape computation, overflowing intermediate int arithmetic and corrupting heap allocation sizes.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for attribute.hpp:121 (CWE-195 signed->unsigned narrowing of
// AttributeProto::ints() into std::vector<std::size_t>). A model whose MaxPool
// kernel_shape contains a negative element (e.g. [3,-1]) currently turns -1 into
// SIZE_MAX silently; after the fix, get_value<vector<size_t>> must reject the
// negative element so convert_model throws ov::Exception instead of producing a
// bogus SIZE_MAX spatial dim.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// TODO: add the crafted fixture models/negative_kernel_shape.onnx (a MaxPool/Conv
//       node with attribute kernel_shape = [3, -1]) under the frontend test
//       models dir; reuse the existing convert_model() helper from this TU.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_kernel_shape_rejected) {
    // Pre-fix: -1 is narrowed to SIZE_MAX at attribute.hpp:121 and no throw here.
    // Post-fix: the non-negative validation makes import fail cleanly.
    EXPECT_THROW(convert_model("negative_kernel_shape.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_model_negative_kernel_shape_rejected*. Requires crafting models/negative_kernel_shape.onnx (MaxPool with kernel_shape=[3,-1]). Pre-fix the test FAILS (no exception; SIZE_MAX dim flows into shape inference, possibly an unrelated validation throw/ASan signed-unsigned overflow report); post-fix it PASSES because attribute.hpp validates each int >= 0 and throws ov::Exception during convert_model.

## Suggested fix
Replace the range-constructor on line 121 with an explicit loop that validates each element:
```cpp
case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
    std::vector<std::size_t> result;
    result.reserve(attribute.ints_size());
    for (int64_t v : attribute.ints()) {
        if (v < 0) OPENVINO_THROW("Attribute element must be non-negative, got: ", v);
        result.push_back(static_cast<std::size_t>(v));
    }
    return result;
}
```


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #228.
