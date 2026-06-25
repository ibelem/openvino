# Security finding #359: Lines 24 and 26 call std::stoull(entry.value()) without any try/cat…

**Summary:** Lines 24 and 26 call std::stoull(entry.value()) without any try/cat…

**CWE IDs:** CWE-1284: Improper Validation of Specified Quantity in Input
**Severity / Impact:** An ONNX model whose external_data 'offset' or 'length' field contains a non-numeric string or an overflowed integer causes an unhandled exception that propagates out of the ONNX frontend loader, crashing the hosting application. This is a reliable remote/local DoS: any untrusted ONNX file with a malformed external_data length field triggers it.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX proto external_data 'offset' and 'length' string fields parsed directly with std::stoull

## Description / Root cause
Lines 24 and 26 call std::stoull(entry.value()) without any try/catch or pre-validation. std::stoull throws std::invalid_argument if the string is not a valid integer (e.g., 'NaN', '', '-1', '0x1') and std::out_of_range if the value exceeds ULLONG_MAX. Neither exception is caught in the constructor, and no wrapping try/catch covers the TensorExternalData constructor call sites in the frontend loading path.

**Validator analysis:** Confirmed by reading tensor_external_data.cpp:19-36: the TensorProto constructor parses external_data string fields and calls std::stoull at lines 24 (offset) and 26 (length) with no surrounding try/catch and no numeric pre-validation. std::stoull throws std::invalid_argument for non-numeric strings ('NaN', '', '0x1') and std::out_of_range for values > ULLONG_MAX. Unlike the load_external_*data methods (lines 46-50, 76-80) which catch std::runtime_error and convert to error::invalid_external_data, the constructor has no such guard. The later bounds checks (lines 53-54, 65) are only reached AFTER a successful parse, so they do not mitigate the parse-time throw. The thrown type is a raw std::exception subclass, NOT ov::Exception, so callers that only catch ov::Exception around read_model would let it escape — making the DoS plausible. CWE-1284 (improper validation of specified quantity) and the DoS impact are accurate. The proposed fix (wrap both stoull calls in try/catch and rethrow as error::invalid_external_data) is correct and sufficient, matching the existing error-handling style in the same file; a regex/length pre-validation is an acceptable alternative. For openvinoEp the defect is real but not reachable: the EP path does not feed raw external_data offset/length string fields into the frontend.

## Exploit / Proof of Concept
Craft an ONNX TensorProto where external_data contains {key:'offset', value:'not_a_number'} or {key:'length', value:'99999999999999999999999999'}. Feed this to any API that parses external-data tensors (e.g., core.read_model). The constructor at line 24 calls std::stoull('not_a_number'), which throws std::invalid_argument; this exception exits the constructor and, if uncaught by the frontend's I/O layer, terminates the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for tensor_external_data.cpp:24,26 (std::stoull on
// attacker-controlled external_data 'offset'/'length' strings, unguarded).
// Pre-fix: convert_model throws std::invalid_argument (not ov::Exception),
//          which EXPECT_THROW(..., ov::Exception) FAILS to match -> test fails.
// Post-fix: constructor catches and rethrows error::invalid_external_data
//          (an ov::Exception subclass) -> EXPECT_THROW matches -> test passes.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture model whose TensorProto.external_data carries
//       {key:"location", value:"data.bin"} and {key:"offset", value:"not_a_number"}
//       (and a sibling case with length="99999999999999999999999999").
//       These binary .onnx fixtures must be added under the onnx frontend test
//       models dir; see TODO below.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_invalid_offset_string) {
    // TODO: add fixture 'external_data_invalid_offset.onnx' under
    //       onnx/frontend/tests/models/ with external_data offset="not_a_number".
    EXPECT_THROW(convert_model("external_data_invalid_offset.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_length_string) {
    // TODO: add fixture 'external_data_overflow_length.onnx' under
    //       onnx/frontend/tests/models/ with external_data length="99999999999999999999999999".
    EXPECT_THROW(convert_model("external_data_overflow_length.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Add the two crafted .onnx fixtures (external_data offset="not_a_number" and length="99999999999999999999999999") under src/frontends/onnx/tests/models/. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_invalid_offset_string*:*onnx_external_data_overflow_length_string*. Pre-fix the constructor throws std::invalid_argument/std::out_of_range (not ov::Exception) so EXPECT_THROW(...,ov::Exception) fails; post-fix it throws error::invalid_external_data (ov::Exception) and the test passes.

## Suggested fix
Wrap both stoull calls in a try/catch block and convert exceptions to error::invalid_external_data: `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& e) { throw error::invalid_external_data(std::string("Invalid offset value: ") + e.what()); }`. Apply the same pattern to the length field on line 26. Alternatively, validate that entry.value() matches `[0-9]+` before calling stoull.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #359.
