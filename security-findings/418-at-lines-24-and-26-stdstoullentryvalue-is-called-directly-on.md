# Security finding #418: At lines 24 and 26, `std::stoull(entry.value())` is called directly…

**Summary:** At lines 24 and 26, `std::stoull(entry.value())` is called directly…

**CWE IDs:** CWE-248: Uncaught Exception / CWE-20: Improper Input Validation
**Severity / Impact:** Denial of service: a crafted ONNX model with `offset="not_a_number"` or `length="99999999999999999999999"` will abort/crash the loading process. Depending on how the caller handles C++ exceptions, this may be an unhandled `std::exception` that terminates the inference-engine process or the host application.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX protobuf external_data string field → std::stoull in TensorExternalData constructor

## Description / Root cause
At lines 24 and 26, `std::stoull(entry.value())` is called directly on the attacker-supplied strings for `"offset"` and `"length"` with no try/catch. If either value is non-numeric, empty, or out of `unsigned long long` range, `std::stoull` throws `std::invalid_argument` or `std::out_of_range`. The constructor provides no exception handler, so these propagate up through the ONNX frontend's model-load path as unhandled C++ exceptions.

**Validator analysis:** The flaw is real and reachable for openvino: the string-taking constructor (line 19) is invoked while parsing a .onnx TensorProto with data_location=EXTERNAL during read_model; offset/length come straight from the file's external_data entries and are fed to std::stoull with no guard. A non-numeric ('AAAA') or oversized ('9999...') value throws std::invalid_argument/std::out_of_range, which is a std::logic_error — NOT an ov::Exception — so it escapes the frontend's domain-error contract; callers expecting ov::Exception will see an unexpected std::exception (DoS / wrong-exception-type). CWE-20/CWE-248 categorisation is accurate; impact is more precisely 'wrong/uncaught exception type leaking from the frontend' than a hard crash (top-level ORT/app catches of std::exception may absorb it, but the frontend should emit invalid_external_data). The proposed fix (wrap each std::stoull in try/catch -> throw error::invalid_external_data{...}) is correct and sufficient and matches the existing error handling style in the same file (e.g. lines 48-50). For openvinoEp the path is not reachable: the EP populates offsets via the size_t constructor (line 37) using ORT_MEM_ADDR, so line 24/26 string parsing is not driven by the EP's untrusted input.

## Exploit / Proof of Concept
Craft an ONNX model with a tensor initializer whose `data_location=EXTERNAL` and `external_data` contains `{key="offset", value="AAAA"}`. When the ONNX frontend's `Tensor::get_data()` path reaches `TensorExternalData(*m_tensor_proto)`, line 24 calls `std::stoull("AAAA")`, which throws `std::invalid_argument`. If the application has no top-level catch for `std::exception`, the process terminates.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor_external_data.cpp:24,26 (std::stoull on unvalidated
// attacker-controlled external_data 'offset'/'length' strings).
//
// Pre-fix: convert_model on a model whose initializer has data_location=EXTERNAL
// and external_data offset="AAAA" (or length="99999999999999999999999") lets
// std::stoull throw std::invalid_argument / std::out_of_range, which is a
// std::logic_error, NOT an ov::Exception -> this EXPECT_THROW(..., ov::Exception)
// FAILS (wrong exception type escapes the frontend).
// Post-fix: the constructor wraps std::stoull and rethrows
// error::invalid_external_data (an ov::Exception) -> assertion passes.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
// NOTE: needs a crafted .onnx fixture (see TODO) -> skeleton.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_bad_offset_string_throws_ov_exception) {
    // TODO: add the crafted model fixture under
    //   onnx/frontend/tests/models/external_data/external_data_bad_offset.onnx
    // containing one initializer with data_location=EXTERNAL and an
    // external_data entry {key="offset", value="AAAA"} (non-numeric).
    // Build the model object the same way other onnx_import.in.cpp tests do, e.g.:
    //   const auto model = convert_model("external_data/external_data_bad_offset.onnx");
    EXPECT_THROW(convert_model("external_data/external_data_bad_offset.onnx"),
                 ov::Exception);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_overflow_length_throws_ov_exception) {
    // TODO: fixture external_data_overflow_length.onnx with external_data entry
    //   {key="length", value="99999999999999999999999"} (> ULLONG_MAX) to drive
    //   the std::out_of_range branch at tensor_external_data.cpp:26.
    EXPECT_THROW(convert_model("external_data/external_data_overflow_length.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_bad_offset_string_throws_ov_exception*:*onnx_external_data_overflow_length_throws_ov_exception*'. Pre-fix the std::stoull at tensor_external_data.cpp:24/26 throws std::invalid_argument/std::out_of_range (a std::logic_error, not ov::Exception) so EXPECT_THROW(..., ov::Exception) fails / the test aborts with an unexpected exception; post-fix it throws error::invalid_external_data (an ov::Exception) and passes. Requires authoring the two crafted .onnx fixtures noted in the TODOs.

## Suggested fix
Wrap the `std::stoull` calls in a try/catch and convert them into a domain-specific `invalid_external_data` exception: `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& e) { throw error::invalid_external_data{"invalid offset: " + std::string(e.what())}; }`. Apply the same pattern to the `length` parsing at line 26.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #418.
