# Security finding #434: Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->in…

**Summary:** Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->in…

**CWE IDs:** CWE-681: Incorrect Conversion between Numeric Types
**Severity / Impact:** Every bfloat16 initializer/constant in an ONNX model loaded via int32_data is silently mis-decoded: the integer 0x3F80 (bfloat16 bit-pattern for 1.0) is converted to the bfloat16 encoding of the float 16256.0. An attacker controlling the model can craft int32_data values to inject arbitrary bfloat16 bit-patterns — including NaN, ±Inf, or specific mantissa patterns — into graph initializers that the model's legitimate author never intended, causing silently wrong inference results or NaN/Inf propagation through the inference graph. This affects any user loading an untrusted ONNX model that uses BFLOAT16 tensors stored as int32_data (not raw_data).
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:148` — `Tensor::get_data<ov::bfloat16>()()`
**Validated for repos:** openvino
**Trust boundary:** External ONNX model file → protobuf int32_data field → __get_data<ov::bfloat16>

## Description / Root cause
Line 148 calls `detail::__get_data<ov::bfloat16>(m_tensor_proto->int32_data())`, which expands (tensor.hpp:50) to `std::vector<ov::bfloat16>(std::begin(int32_data), std::end(int32_data))`. This invokes the `ov::bfloat16` *numeric* constructor with each raw int32 value, treating the stored 16-bit bit-pattern as an integer magnitude. The ONNX spec mandates that bfloat16 values are stored as raw 16-bit IEEE bit-patterns in the low bits of int32_data — identical to float16. The float16 path (lines 123-124) correctly uses `ov::float16::from_bits(static_cast<uint16_t>(elem))` to reinterpret the bits; the bfloat16 path has no equivalent `from_bits` call and performs a numeric conversion instead.

**Validator analysis:** Confirmed real: tensor.cpp:147-148 gates on TensorProto_DataType_BFLOAT16 then calls detail::__get_data<ov::bfloat16>(m_tensor_proto->int32_data()). The single-arg __get_data template (tensor.hpp:43-54) constructs std::vector<ov::bfloat16>(std::begin, std::end), which numerically converts each int32 magnitude into bfloat16, NOT a bit reinterpretation. The float16 sibling path (tensor.cpp:116-128) correctly uses ov::float16::from_bits(static_cast<uint16_t>(elem)). Per ONNX, BFLOAT16 in int32_data is a raw 16-bit bit-pattern, so this is a genuine CWE-681 mis-decode: e.g. 0x3F80 (1.0f bf16) becomes bf16(16256.0). vulnType CWE-681 is accurate; impact is correctness/silent-wrong-results (NaN/Inf propagation), NOT a memory-safety/RCE issue, so severity should be framed as integrity/correctness rather than crash. The proposed fix is correct and sufficient: it mirrors the float16 branch using ov::bfloat16::from_bits(static_cast<uint16_t>(elem)) over int32_data, matching ONNX semantics. Note the analogous int32_data path for FLOAT16 is already correct; only the BFLOAT16 int32_data branch (line 148) is wrong (the raw_data branch at 145 and the m_tensor_place raw branch at 139 are byte-copies and fine). openvinoEp marked rejected because reachability of the int32_data (vs raw_data) branch through ORT's subgraph serialization was not confirmed within budget.

## Exploit / Proof of Concept
Supply an ONNX model with a BFLOAT16 tensor whose int32_data list contains values like [0x3F80, 0xFF80, 0x7FC0] (bfloat16 bit-patterns for 1.0, -Inf, NaN). Because line 148 feeds these raw integers into `ov::bfloat16`'s numeric constructor, the resulting vector will contain the bfloat16 encodings of the integers 16256, 65408, 32704 respectively — entirely different values. Conversely, an attacker who wants to inject bfloat16 NaN (bit-pattern 0x7FC0 = 32704) into a weight tensor simply stores int32_data[i] = 32704, which the numeric constructor will round to the nearest representable bfloat16, potentially landing on a NaN encoding. No authentication, schema validation, or bounds check anywhere on this path prevents the misdecoding — the BFLOAT16 data_type check at line 147 only gates the path, it does not validate the values.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:148.
// Pre-fix: __get_data<ov::bfloat16>(int32_data()) numerically converts the stored int32
//          magnitude (e.g. 0x3F80 -> bf16(16256.0)) instead of reinterpreting the low 16
//          bits as a bfloat16 bit-pattern.
// Post-fix: int32_data entries are decoded with ov::bfloat16::from_bits, so 0x3F80 -> 1.0f.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// This needs a crafted .onnx whose BFLOAT16 initializer stores values in int32_data
// (NOT raw_data), so the data lands on tensor.cpp line 147-148. Hence a binary fixture
// is required and the symbol/path details below are best-effort TODOs.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm include used by onnx_import.in.cpp for convert_model()
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace/helper for convert_model

TEST(onnx_import_bfloat16, int32_data_bitpattern_decode) {
    // TODO: provide models/bfloat16_int32_data_const.onnx:
    //   - a single BFLOAT16 Constant/initializer of shape [3]
    //   - data stored in int32_data = {0x3F80, 0xC000, 0x4040}
    //     (bf16 bit-patterns for 1.0, -2.0, 3.0), data_type = BFLOAT16, no raw_data.
    const auto model = convert_model("bfloat16_int32_data_const.onnx");

    std::shared_ptr<ov::op::v0::Constant> cst;
    for (const auto& node : model->get_ordered_ops()) {
        if ((cst = ov::as_type_ptr<ov::op::v0::Constant>(node))) break;
    }
    ASSERT_NE(cst, nullptr);
    ASSERT_EQ(cst->get_element_type(), ov::element::bf16);

    const auto values = cst->cast_vector<float>();
    ASSERT_EQ(values.size(), 3u);
    // Pre-fix these are 16256.0, -49152.0, 49216.0 (numeric ctor); fix yields the bit-patterns.
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], -2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='onnx_import_bfloat16.int32_data_bitpattern_decode'. Requires adding fixture models/bfloat16_int32_data_const.onnx (BFLOAT16 initializer with int32_data {0x3F80,0xC000,0x4040}, no raw_data). Pre-fix: assertion fails because decoded values are the numeric conversions (~16256.0 etc.) rather than 1.0/-2.0/3.0; no sanitizer error (this is a correctness, not memory-safety, defect). Post-fix (from_bits over int32_data): assertion passes.

## Suggested fix
Apply the same bit-reinterpretation pattern used by the float16 branch. Replace line 148 with:

```cpp
if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
    using std::begin; using std::end;
    const auto& int32_data = m_tensor_proto->int32_data();
    std::vector<ov::bfloat16> bf16_data;
    bf16_data.reserve(int32_data.size());
    std::transform(begin(int32_data), end(int32_data), std::back_inserter(bf16_data),
        [](int32_t elem) { return ov::bfloat16::from_bits(static_cast<uint16_t>(elem)); });
    return bf16_data;
}
```

This matches the float16 handling at lines 120-127 and correctly reinterprets the low 16 bits of each int32_data entry as a bfloat16 bit-pattern without any numeric conversion.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #434.
