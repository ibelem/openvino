# Security finding #438: Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->in…

**Summary:** Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->in…

**CWE IDs:** CWE-681: Incorrect Conversion between Numeric Types
**Severity / Impact:** Silent data corruption of all bfloat16 model weights and constants loaded via the non-raw ONNX protobuf path. Downstream constant-folding (including ov::op::v1::Divide at div.cpp:23) operates on wrong values. An attacker supplying a crafted ONNX model can systematically misdirect inference, potentially causing incorrect security-relevant decisions, and can craft specific int32 input values to produce attacker-chosen bfloat16 bit patterns (including Inf/NaN) in the computed constant after downstream arithmetic operations.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:148` — `Tensor::get_data<ov::bfloat16>()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model file → TensorProto int32_data field for BFLOAT16 initializers → Tensor::get_data<ov::bfloat16>()

## Description / Root cause
Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->int32_data())` which constructs `ov::bfloat16` from each `int32_t` via implicit numeric conversion (integer-to-float cast). The ONNX spec stores bfloat16 values as their 16-bit IEEE bit patterns in the lower 16 bits of each int32 field, requiring `ov::bfloat16::from_bits(static_cast<uint16_t>(elem))`. The float16 path at lines 120-128 correctly uses `from_bits`; the bfloat16 path does not. Every non-zero bfloat16 constant loaded from a non-raw ONNX protobuf is silently corrupted: the int32 bit pattern (e.g. 0x3F80 = 16256 for bfloat16 1.0) is re-interpreted as the integer 16256 and converted numerically to bfloat16, producing bfloat16(16256) ≈ 16256.0 instead of 1.0.

**Validator analysis:** Confirmed real. In tensor.hpp:43-54, __get_data<T,Container> does std::vector<T>(begin,end), invoking ov::bfloat16's numeric (float) constructor on each int32_t. The ONNX spec stores bfloat16 as the 16-bit IEEE bit pattern in the low bits of int32_data, so the correct decode is bfloat16::from_bits(static_cast<uint16_t>(elem)) — exactly what the FLOAT16 branch at lines 120-128 does. The BFLOAT16 branch (line 148) and the m_tensor_place branch (line 142, __get_data<ov::bfloat16,int32_t>) both lack this, so non-raw bfloat16 initializers are silently reinterpreted (0x3F80=16256 becomes ~16256.0 instead of 1.0). vulnType CWE-681 (Incorrect Numeric Conversion) is accurate; impact is silent data corruption of bfloat16 weights — accurate, though the 'security-relevant decisions / attacker-chosen NaN' framing is speculative since this is functional corruption, not memory unsafety. The proposed fix is correct and matches the existing float16 pattern, but is INCOMPLETE: it only fixes line 148. The identical bug at line 142 (m_tensor_place non-raw path, __get_data<ov::bfloat16,int32_t>) must also be converted to from_bits decoding; otherwise the place-based loader still corrupts. Recommend applying the from_bits transform to BOTH branches.

## Exploit / Proof of Concept
Craft an ONNX model with a BFLOAT16-typed initializer whose int32_data fields contain the raw bit pattern of bfloat16 1.0 (0x3F80 = int32 16256). When loaded through get_data<ov::bfloat16>() at line 148, the numeric conversion ov::bfloat16(16256) produces bfloat16(16256.0f) instead of 1.0f. This constant is then wrapped in ov::op::v0::Constant at tensor.cpp:518 and fed, e.g., as a denominator in a Divide node; the graph executes with the wrong constant value throughout constant-folding.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for tensor.cpp:148 (and the twin defect at line 142):
//   non-raw BFLOAT16 initializers stored as 16-bit bit patterns in int32_data
//   must decode via ov::bfloat16::from_bits, not numeric int->float conversion.
// Pre-fix: the constant value reads ~16256.0 (0x3F80 reinterpreted numerically)
//   instead of 1.0 -> assertion fails.
// Post-fix: from_bits decode yields 1.0 -> assertion passes.
// Harness: ov_onnx_frontend_tests, TEST in the style of onnx_import.in.cpp.
//
// NOTE (skeleton): this needs a crafted .prototxt fixture under the onnx test
// models dir declaring a BFLOAT16 Constant/initializer whose int32_data element
// = 16256 (0x3F80 == bfloat16 1.0). Symbol/path TODOs flagged below.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm helper header name in this test dir
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace

OPENVINO_TEST(${FRONTEND_NAME}_onnx, model_bfloat16_initializer_nonraw_from_bits) {
    // TODO: create models/bfloat16_const_nonraw.prototxt:
    //   a graph with one BFLOAT16 initializer/Constant, dims=[1],
    //   int32_data: 16256   (0x3F80 == bit pattern of bfloat16 1.0)
    const auto model = convert_model("bfloat16_const_nonraw.prototxt");

    std::shared_ptr<op::v0::Constant> cst;
    for (const auto& node : model->get_ordered_ops()) {
        if ((cst = ov::as_type_ptr<op::v0::Constant>(node))) break;
    }
    ASSERT_NE(cst, nullptr);
    ASSERT_EQ(cst->get_element_type(), element::bf16);

    const auto values = cst->cast_vector<float>();
    ASSERT_EQ(values.size(), 1u);
    // Pre-fix this is ~16256.0f; the fix must decode the bit pattern to 1.0f.
    EXPECT_FLOAT_EQ(values[0], 1.0f);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*model_bfloat16_initializer_nonraw_from_bits*. Requires authoring a crafted bfloat16_const_nonraw.prototxt fixture (BFLOAT16 initializer, int32_data=16256). Expected: pre-fix the constant reads ~16256.0 and EXPECT_FLOAT_EQ fails; post-fix (from_bits decode) it reads 1.0 and passes. No sanitizer error expected — this is functional data-corruption, not memory unsafety.

## Suggested fix
Replace line 148 with the same pattern used for float16 (lines 120-128): iterate over int32_data and call `ov::bfloat16::from_bits(static_cast<uint16_t>(elem))` for each element. For example: `const auto& int32_data = m_tensor_proto->int32_data(); std::vector<ov::bfloat16> bf16_data; bf16_data.reserve(int32_data.size()); std::transform(begin(int32_data), end(int32_data), std::back_inserter(bf16_data), [](int32_t elem){ return ov::bfloat16::from_bits(static_cast<uint16_t>(elem)); }); return detail::__get_data<ov::bfloat16>(bf16_data);`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #438.
