# Security finding #473: At lines 24 and 26, `std::stoull(entry.value())` is called directly…

**Summary:** At lines 24 and 26, `std::stoull(entry.value())` is called directly…

**CWE IDs:** CWE-248: Uncaught Exception / CWE-20: Improper Input Validation
**Severity / Impact:** An adversary who can supply a crafted ONNX model file can terminate model-loading with an unhandled C++ exception, causing an abrupt process abort or propagating through the inference-engine stack in an uncontrolled manner. In a server-side inference endpoint this is a reliable remote DoS; in an offline conversion tool it is an application crash. It also prevents callers from distinguishing invalid-data errors from other runtime errors.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX protobuf external_data fields (location/offset/length) parsed in TensorExternalData constructor

## Description / Root cause
At lines 24 and 26, `std::stoull(entry.value())` is called directly on attacker-controlled protobuf string values for 'offset' and 'length' with no surrounding try/catch and no pre-validation. `std::stoull` throws `std::invalid_argument` if the string is non-numeric (e.g. empty string, letters, '+', '-') and `std::out_of_range` if the value exceeds `ULLONG_MAX`. Neither exception is caught inside the constructor, nor by any catch in the immediate callers (`load_external_data`/`load_external_mmap_data`), whose try/catch blocks only cover `std::runtime_error`.

**Validator analysis:** The flaw is real: at tensor_external_data.cpp:24 and :26 the constructor parses untrusted ONNX protobuf strings with std::stoull and no validation; std::stoull throws std::invalid_argument on non-numeric input ('abc', '', '+', '-') and std::out_of_range on >ULLONG_MAX. These are std::logic_error subclasses, NOT std::runtime_error, so the only nearby try/catch blocks (load_external_data:78 / load_external_mmap_data:48, which catch std::runtime_error from sanitize_path) do not apply — and crucially those run AFTER the constructor; the constructor itself runs in get_external_data() (tensor.hpp:318-322) with no surrounding handler. The CWE-20/CWE-248 categorisation is accurate. The 'remote process abort' impact is somewhat overstated: ov::Core::read_model generally catches std::exception and rethrows as ov::Exception, so in the normal OV path this surfaces as a thrown exception rather than a true abort(); the concrete defect is that invalid external-data fields produce an undescriptive logic_error instead of the project's error::invalid_external_data, violating the documented error contract and risking an unhandled exception in any caller that only catches the documented type. The proposed fix (wrap both stoull calls in try/catch converting to error::invalid_external_data, plus a digits-only/non-empty precheck) is correct and sufficient; it should also reject negative-sign strings that stoull silently wraps. Validated for openvino only.

## Exploit / Proof of Concept
Craft a .onnx file whose TensorProto contains an external_data entry with key='offset' and value='abc' (or value='99999999999999999999999999999'). When OpenVINO's ONNX front-end deserialises the model, `TensorExternalData::TensorExternalData` is called, `std::stoull("abc")` throws `std::invalid_argument`, the exception propagates past the constructor with no handler and ultimately aborts or corrupts the inference stack.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (TensorExternalData ctor std::stoull on
// untrusted external_data 'offset'/'length'). Pre-fix: convert_model on a model whose external_data
// 'offset' (or 'length') value is non-numeric throws std::invalid_argument (uncaught logic_error),
// failing EXPECT_THROW(..., ov::Exception). Post-fix it is converted to error::invalid_external_data
// (an ov::Exception), so the assertion passes.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
// NOTE: requires a crafted fixture model (see TODO) because TensorExternalData is constructed from a
// TensorProto deep inside Tensor::get_external_data (tensor.hpp:318-322) reached only via convert_model.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: add models/external_data/invalid_offset_nonnumeric.onnx + .data where a TensorProto has
//       data_location=EXTERNAL and external_data entry key="offset", value="abc" (and a sibling
//       fixture with key="length", value="99999999999999999999999999999" for the out_of_range case).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_invalid_offset_string) {
    EXPECT_THROW(convert_model("external_data/invalid_offset_nonnumeric.onnx"), ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_length_out_of_range) {
    EXPECT_THROW(convert_model("external_data/invalid_length_overflow.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_invalid_offset_string*:*onnx_external_data_length_out_of_range*'. Pre-fix expectation: test FAILS because std::stoull throws std::invalid_argument/std::out_of_range (a std::logic_error) which is not an ov::Exception (or under ASan/uncaught-exception harness aborts); post-fix the ctor converts it to error::invalid_external_data (ov::Exception) and the test passes. Requires the two crafted .onnx fixtures noted in the TODO.

## Suggested fix
Wrap both `stoull` calls in a try/catch that converts exceptions to a descriptive `std::runtime_error` or the project's `error::invalid_external_data` type. For example: `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& e) { throw error::invalid_external_data{"invalid 'offset' value: " + entry.value()}; }` Apply the same pattern for 'length'. Additionally, add a pre-check that the value string is non-empty and contains only decimal digits before calling `stoull`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #473.
