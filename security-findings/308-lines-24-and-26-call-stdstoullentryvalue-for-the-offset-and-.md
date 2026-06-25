# Security finding #308: Lines 24 and 26 call `std::stoull(entry.value())` for the 'offset' …

**Summary:** Lines 24 and 26 call `std::stoull(entry.value())` for the 'offset' …

**CWE IDs:** CWE-248: Uncaught Exception / CWE-20: Improper Input Validation
**Severity / Impact:** Any process that loads a crafted ONNX model through the OpenVINO ONNX frontend terminates immediately with an uncaught C++ exception. This is a reliable, zero-effort denial-of-service: a single malformed field (`offset: ""`, `length: "abc"`, or `offset: "99999999999999999999999999"`) is sufficient to crash the inference engine — and any containing application — regardless of further bounds checks in `load_external_data` / `load_external_mmap_data`.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** Externally supplied ONNX model file → TensorProto::external_data() string fields → TensorExternalData constructor

## Description / Root cause
Lines 24 and 26 call `std::stoull(entry.value())` for the 'offset' and 'length' keys of `tensor.external_data()` from an attacker-controlled ONNX model, with no try/catch block anywhere in the constructor body and no prior format/range validation of the string. `std::stoull` throws `std::invalid_argument` for empty or non-numeric strings and `std::out_of_range` for values exceeding ULLONG_MAX. Neither exception is caught here, and inspection of the call chain in `frontend.cpp` (`load_impl` at line 170, `convert` at line 310) shows no surrounding try/catch that would trap `std::invalid_argument` or `std::out_of_range` before they escape the frontend.

**Validator analysis:** Confirmed real: TensorExternalData ctor (lines 19-36) parses 'offset'/'length' from tensor.external_data() via std::stoull with zero validation and no surrounding try/catch, unlike the rest of the file which uses error::invalid_external_data (e.g. lines 49,55). A crafted ONNX initializer with data_location EXTERNAL and offset="" or length="abc" yields std::invalid_argument; a 26-digit value yields std::out_of_range. This is reachable during ONNX model conversion. The vulnType is best framed as CWE-20 / CWE-248: at minimum it leaks a non-domain std exception instead of a catchable OpenVINO error; whether it reaches std::terminate (full DoS) depends on the caller — many frontend callers catch std::exception, so the 'reliable process termination' impact is overstated, but the input-validation defect is genuine and worth fixing. The proposed fix (wrap each stoull in try/catch and rethrow as error::invalid_external_data) is correct and sufficient; equivalently validate the digit string before conversion. For the openvinoEp boundary the malformed string must survive ORT's own protobuf parsing and the defect is not in EP-owned code, so rejected there.

## Exploit / Proof of Concept
Craft an ONNX model whose initializer tensor has `data_location: EXTERNAL` and an `external_data` entry with key `offset` and value `""` (empty string) or `"not_a_number"`. When the ONNX frontend calls `TensorExternalData::TensorExternalData(tensor)`, line 24 executes `std::stoull("")` which throws `std::invalid_argument`. Because neither the constructor nor any frame in the `load_impl`→`InputModel`→`TensorExternalData` call chain catches `std::invalid_argument` or `std::logic_error`, the exception propagates to the application boundary uncaught, terminating the process via `std::terminate`.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor_external_data.cpp:24,26 (TensorExternalData ctor).
// Pre-fix: std::stoull on a malformed offset/length string ('' or 'abc' or a
//   26-digit overflow) throws std::invalid_argument / std::out_of_range, a raw
//   std exception that is NOT an ov::Exception -> EXPECT_THROW(..., ov::Exception)
//   FAILS pre-fix (wrong exception type escapes).
// Post-fix: ctor wraps stoull and rethrows error::invalid_external_data (an
//   ov::Exception subclass) -> assertion passes.
//
// Style mirrors onnx_import.in.cpp; uses convert_model() helper.
// SKELETON: requires a crafted ONNX fixture with an initializer whose
//   data_location=EXTERNAL and external_data entry key='offset' value=''.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: create fixture models/external_data/bad_offset_empty.onnx with an
//       initializer: data_location=EXTERNAL, external_data{key:'offset',value:''}
//       (and similarly bad_length_nan.onnx with key:'length',value:'abc',
//        and bad_offset_overflow.onnx with value:'99999999999999999999999999').
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_offset_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_offset_empty.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_length_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_length_nan.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_offset_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_offset_overflow.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_malformed*:*onnx_external_data_overflow*'. Pre-fix the tests FAIL because std::stoull raises std::invalid_argument/std::out_of_range (a non-ov::Exception std exception escapes convert_model); post-fix the ctor rethrows error::invalid_external_data (ov::Exception) and the EXPECT_THROW passes. Requires crafting the three .onnx fixtures noted in the TODO.

## Suggested fix
Wrap each `std::stoull` call in a try/catch and rethrow as an OpenVINO domain error, e.g.:
```cpp
} else if (entry.key() == "offset") {
    try {
        m_offset = std::stoull(entry.value());
    } catch (const std::exception& e) {
        throw error::invalid_external_data{
            "Invalid 'offset' value '" + entry.value() + "': " + e.what()};
    }
} else if (entry.key() == "length") {
    try {
        m_data_length = std::stoull(entry.value());
    } catch (const std::exception& e) {
        throw error::invalid_external_data{
            "Invalid 'length' value '" + entry.value() + "': " + e.what()};
    }
```
Alternatively, validate the string with a regex or manual digit-check before conversion. Using `error::invalid_external_data` (already used elsewhere in the same file) ensures the error propagates as a known, catchable OpenVINO exception rather than an unhandled C++ standard exception.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #308.
