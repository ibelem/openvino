# Security finding #91: Lines 24 and 26 call `std::stoull(entry.value())` bare, with no sur…

**Summary:** Lines 24 and 26 call `std::stoull(entry.value())` bare, with no sur…

**CWE IDs:** CWE-248: Uncaught Exception / CWE-20: Improper Input Validation
**Severity / Impact:** Denial-of-service: loading any attacker-supplied ONNX model whose `external_data` block contains a non-numeric or overflow-inducing `offset` or `length` string causes an uncaught C++ exception that unwinds past `FrontEnd::convert`, terminating or crashing the hosting process. Affects any application that passes user-supplied or network-fetched ONNX models to the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX protobuf external_data map entries ('offset' and 'length' string fields) parsed from an attacker-supplied .onnx model file

## Description / Root cause
Lines 24 and 26 call `std::stoull(entry.value())` bare, with no surrounding try/catch. `std::stoull` throws `std::invalid_argument` (for non-numeric strings like "abc" or "") or `std::out_of_range` (for strings exceeding ULLONG_MAX, e.g. "99999999999999999999999"). The constructor carries no exception handler, and neither caller — `Tensor::get_external_data()` (tensor.hpp:322) nor `Tensor::get_data_size()` (tensor.hpp:372) — wraps the constructor call in a try/catch. The outer convert call at `frontend.cpp:331` likewise has no catch for `std::logic_error` or `std::out_of_range`. The existing `catch (const std::runtime_error& e)` guards in `load_external_mmap_data`/`load_external_data` are irrelevant because the throw happens in the constructor, before those functions are entered.

**Validator analysis:** The flaw is real and confirmed by reading the code: lines 24/26 invoke std::stoull(entry.value()) with no exception handling, while the entry.value() strings come straight from the untrusted ONNX protobuf external_data map. std::stoull throws std::invalid_argument on non-numeric input and std::out_of_range on overflow (>ULLONG_MAX). The two existing try/catch blocks in this file are in load_external_mmap_data (lines 46-50) and load_external_data (lines 76-80) and only guard sanitize_path — they are entered AFTER construction, so they cannot catch the constructor's throw. The constructor is reached during model conversion through Tensor::get_external_data() (tensor.hpp:322) and Tensor::get_data_size() (tensor.hpp:372), both invoked from get_ov_constant on tensors flagged TensorProto_DataLocation_EXTERNAL, and ultimately from FrontEnd::convert (frontend.cpp:331) — none of which wrap the call. The vuln type CWE-248/CWE-20 is accurate (uncaught exception from improper input validation). Impact is best characterized as a robustness/DoS issue: the exception type is std::logic_error rather than the repo's own error::invalid_external_data, so any caller that only catches ov::Exception (or none at all, e.g. a CLI/benchmark host) terminates; a well-written host catching std::exception would survive, so 'crash' is the worst case, not guaranteed. The proposed fix is correct and idiomatic — wrapping each std::stoull in try/catch and rethrowing as error::invalid_external_data matches how every other failure in this file is reported and is handled gracefully by the rest of the frontend. It is sufficient; one refinement: catch std::exception (covers both invalid_argument and out_of_range) and include which field ('offset'/'length') failed, exactly as the proposal does. Note the actual trigger requires a crafted .onnx binary fixture, so the regression test is a skeleton.

## Exploit / Proof of Concept
Craft an ONNX model (protobuf) where a tensor's `external_data` repeated field contains `{key: "offset", value: "not_a_number"}` or `{key: "length", value: "99999999999999999999999999"}`. When the model is loaded via `FrontEnd::load` + `FrontEnd::convert`, the `TensorExternalData` constructor is invoked, `std::stoull` throws `std::invalid_argument` or `std::out_of_range`, and the exception propagates uncaught, crashing the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for unchecked std::stoull at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24 and :26
// (TensorExternalData::TensorExternalData(const TensorProto&)).
//
// What this encodes:
//   A model whose tensor external_data map has a non-numeric 'offset'
//   (e.g. value="not_a_number") or an overflowing 'length'
//   (e.g. value="99999999999999999999999999") must be REJECTED with a
//   frontend ov::Exception (error::invalid_external_data), NOT crash the
//   process with an uncaught std::invalid_argument / std::out_of_range.
//
//   Pre-fix: convert_model throws std::out_of_range/std::invalid_argument
//            (a std::logic_error, not ov::Exception) -> EXPECT_THROW(...,
//            ov::Exception) FAILS (wrong exception type escapes).
//   Post-fix: the constructor rethrows error::invalid_external_data, which
//            derives from ov::Exception -> assertion passes.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_malformed_offset_length_rejected) {
    // TODO: provide a crafted ONNX fixture under
    //   src/frontends/onnx/tests/models/external_data/
    // whose initializer has data_location=EXTERNAL and an external_data entry
    //   {key:"offset", value:"not_a_number"}  (or length:"99999999999999999999999999").
    // This binary/protobuf fixture cannot be authored as plain text here, hence
    // the value below is a placeholder file name.
    EXPECT_THROW(convert_model("external_data/external_data_malformed_offset.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_malformed_offset_length_rejected*. Expected pre-fix: test FAILS because an uncaught std::invalid_argument ("stoull") / std::out_of_range propagates instead of ov::Exception, and under ASan/uncaught-exception handling the process aborts (terminate called after throwing an instance of 'std::out_of_range'). Expected post-fix: TensorExternalData ctor rethrows error::invalid_external_data (an ov::Exception), the EXPECT_THROW matches, and the test passes. TODO: add the crafted .onnx fixture noted in the test before this can compile/run.

## Suggested fix
Wrap each `std::stoull` call in the constructor in a try/catch and rethrow as the existing domain exception type: `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& e) { throw error::invalid_external_data{"Invalid 'offset' value: " + std::string(e.what())}; }` — and identically for `m_data_length` on line 26. This converts any malformed string into a recoverable `invalid_external_data` exception that the rest of the frontend already handles gracefully.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #91.
