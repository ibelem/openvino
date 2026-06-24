# Security finding #229: The INTS branch at line 121 constructs a `std::vector<std::size_t>`…

**Summary:** The INTS branch at line 121 constructs a `std::vector<std::size_t>`…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error
**Severity / Impact:** An attacker who controls a parsed ONNX model can supply a negative value (e.g., -1) in the `kernel_shape`, `strides`, or `dilations` repeated-int field. This wraps to a near-`SIZE_MAX` `size_t` value and is forwarded directly to `ov::Shape` and `ov::Strides` constructors, and subsequently into OpenVINO convolution node constructors. Downstream shape-computation arithmetic using these huge dimension values can integer-overflow into small or zero extents, leading to under-allocated buffers and out-of-bounds writes, or alternatively trigger enormous memory-allocation attempts causing process crash (DoS). Any application that loads untrusted ONNX models through the OpenVINO ONNX frontend is affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/attribute.hpp:121` — `detail::attribute::get_value<std::vector<std::size_t>>()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model protobuf repeated int64 field (AttributeProto::ints()) → std::vector<std::size_t> used as ov::Shape / ov::Strides in convpool.cpp

## Description / Root cause
The INTS branch at line 121 constructs a `std::vector<std::size_t>` via range-initialization from `attribute.ints()`, which is a `repeated int64` protobuf field. The element-wise conversion from `int64_t` to `std::size_t` is implicit and silently wraps negative values: e.g., `-1` becomes `SIZE_MAX` (0xFFFFFFFFFFFFFFFF on 64-bit). No per-element negativity or range check exists anywhere on the path from `attribute.ints()` to the `ov::Shape`/`ov::Strides` constructors called in convpool.cpp lines 26-30 and 59-69.

**Validator analysis:** The flaw is real: at attribute.hpp:115-125 the get_value<std::vector<std::size_t>> INTS branch (line 121) brace-initializes a size_t vector from a repeated int64 field, an implicit narrowing where a negative int64 (e.g. -1) silently wraps to ~SIZE_MAX. The single-INT branch (line 119) and the scalar overload (line 112) have the same defect. convpool::get_kernel_shape (cpp:29), get_strides/get_dilations (cpp:72-77 via cpp:63) call get_attribute_value<std::vector<size_t>> with no per-element validation, so the wrapped value reaches ov::Shape/ov::Strides constructors directly from an attacker-controlled model. The CWE-195 signed-to-unsigned categorization is accurate. The OOB-write impact is overstated/speculative: an enormous kernel/stride dimension most likely triggers a NodeValidationFailure/OPENVINO_ASSERT in Convolution shape inference (graceful exception) or an oversized-allocation DoS, rather than a silent under-allocated buffer; the realistic impact is DoS / unvalidated-input. The proposed fix (per-element OPENVINO_ASSERT(v>=0) before push_back, plus guarding the scalar overload at line 112) is correct and sufficient to close the conversion defect; the optional <2^24 spatial cap is a reasonable hardening but not strictly required since downstream shape inference still validates against actual tensor extents. Apply the same guard to line 119.

## Exploit / Proof of Concept
Craft an ONNX Conv model with `kernel_shape: [-1]` in the node's attributes. When the ONNX frontend parses this, `attribute.ints()` yields `{-1}` (int64). The range-init at line 121 silently converts this to `{SIZE_MAX}`. `convpool::get_kernel_shape` (convpool.cpp:29) returns a `ov::Shape` containing `SIZE_MAX`. This SIZE_MAX-dimensioned shape flows into the convolution node constructor; subsequent shape arithmetic (e.g., output-size = (input - kernel + 2*pad) / stride) overflows, allocating a wrongly-sized output buffer and corrupting memory or crashing.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for attribute.hpp:121 (CWE-195 signed->unsigned conversion):
// a negative value in a repeated-int attribute (kernel_shape/strides/dilations)
// must be rejected, not silently wrapped to ~SIZE_MAX and forwarded to ov::Shape.
// Pre-fix: convert_model deserializes -1 as SIZE_MAX -> huge ov::Shape (no throw /
// later oversized-alloc or assert). Post-fix: per-element OPENVINO_ASSERT(v>=0)
// makes convert_model throw ov::Exception at frontend time.
//
// Harness: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture models/conv_negative_kernel_shape.onnx with a
// Conv node attribute kernel_shape:[-1] (repeated int64). Skeleton below.

OPENVINO_TEST(${BACKEND_NAME}, onnx_conv_negative_kernel_shape_rejected) {
    // TODO: add crafted fixture onnx_import/models/conv_negative_kernel_shape.onnx
    //       containing a Conv node with attribute kernel_shape = [-1] (AttributeProto INTS).
    //       Build it via onnx.helper.make_attribute("kernel_shape", [-1]).
    EXPECT_THROW(convert_model("conv_negative_kernel_shape.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_conv_negative_kernel_shape_rejected*'. Requires crafted fixture conv_negative_kernel_shape.onnx (Conv with kernel_shape=[-1]). Pre-fix: no ov::Exception thrown at frontend deserialization (the -1 wraps to SIZE_MAX), so EXPECT_THROW fails (or an ASan/oversized-allocation abort surfaces downstream); post-fix (OPENVINO_ASSERT(v>=0) in attribute.hpp get_value<std::vector<std::size_t>>): convert_model throws ov::Exception and the test passes.

## Suggested fix
Add a per-element validation loop before constructing the output vector. Replace line 121 with:
```cpp
case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
    std::vector<std::size_t> result;
    result.reserve(attribute.ints_size());
    for (const auto v : attribute.ints()) {
        OPENVINO_ASSERT(v >= 0, "Attribute INTS element must be non-negative, got: ", v);
        result.push_back(static_cast<std::size_t>(v));
    }
    return result;
}
```
Apply the same guard to the single-value `get_value<std::size_t>` overload at line 112 (`OPENVINO_ASSERT(attribute.i() >= 0, ...)`). Optionally add an upper-bound cap (e.g., ≤ 1<<24) for spatial dimensions consumed by convpool.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #229.
