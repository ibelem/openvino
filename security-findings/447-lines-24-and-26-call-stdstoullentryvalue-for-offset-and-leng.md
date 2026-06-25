# Security finding #447: Lines 24 and 26 call `std::stoull(entry.value())` for `offset` and …

**Summary:** Lines 24 and 26 call `std::stoull(entry.value())` for `offset` and …

**CWE IDs:** CWE-248: Uncaught Exception
**Severity / Impact:** Same DoS-by-termination as the graph_iterator_proto case; this constructor path is exercised when the legacy (non-iterator) InputModel path is taken (when `ONNX_ITERATOR` env var is disabled or the model is passed via `istream`).
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model protobuf external_data entries → std::stoull with no input validation

## Description / Root cause
Lines 24 and 26 call `std::stoull(entry.value())` for `offset` and `length` keys respectively, with no surrounding try/catch in the constructor body or any calling context. A non-numeric or out-of-range value in an attacker-controlled ONNX model raises `std::invalid_argument` or `std::out_of_range` that propagates uncaught from the constructor.

**Validator analysis:** The flaw is real: lines 24 and 26 of tensor_external_data.cpp convert the protobuf string values for 'offset'/'length' with std::stoull and have no exception handling, while every other failure path in this very file throws error::invalid_external_data (lines 49,55,66,85,92,118,124). A non-numeric ('abc') or overflowing value in an attacker-supplied ONNX model's external_data raises std::invalid_argument/std::out_of_range that is NOT an ov::Exception, so callers expecting the frontend's documented ov::Exception contract do not catch it — confirmed reachable from the legacy InputModel/istream path via Tensor::get_external_data (tensor.hpp:322) which calls TensorExternalData(*m_tensor_proto). The CWE-248 categorization is accurate; the 'crash/terminate' impact is somewhat overstated (most embedders catch std::exception), but the contract-violation/uncaught-non-ov-exception defect is genuine. The proposed fix (wrap stoull in a helper that rethrows error::invalid_external_data) is correct and matches the file's existing error idiom; it should be applied to BOTH the offset and length parses. Only the openvino repo is reachable here; the defect is openvino-frontend-internal so the EP entry is not on this path.

## Exploit / Proof of Concept
Supply an ONNX model with external tensor data containing key `"offset"` with value `"abc"`. When parsed via the legacy path, `TensorExternalData::TensorExternalData` is called, `std::stoull` throws `std::invalid_argument`, and the exception propagates uncaught through the constructor to the loading call site, crashing the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (CWE-248 uncaught std::stoull).
// Pre-fix: a non-numeric external_data 'offset' value makes the ctor throw
// std::invalid_argument (NOT ov::Exception) -> EXPECT_THROW(..., ov::Exception) fails.
// Post-fix: ctor rethrows error::invalid_external_data (derives from ov::Exception) -> passes.
//
// NOTE: triggering the path requires a crafted ONNX model whose initializer has
// data_location=EXTERNAL and an external_data entry key="offset", value="abc".
// That is a binary protobuf fixture, so this is a SKELETON with TODOs.

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_nonnumeric_offset_throws_ov_exception) {
    // TODO: provide a crafted model fixture under the onnx frontend models dir, e.g.
    //   models/external_data/external_data_bad_offset.onnx
    // whose tensor has:
    //   data_location: EXTERNAL
    //   external_data { key: "location" value: "data.bin" }
    //   external_data { key: "offset"   value: "abc" }   // non-numeric -> std::stoull throws
    // Use the same convert_model helper as onnx_import.in.cpp.
    EXPECT_THROW(convert_model("external_data/external_data_bad_offset.onnx"),
                 ov::Exception);
    // TODO: add an out-of-range variant (value larger than UINT64_MAX) asserting the
    // same ov::Exception (std::out_of_range pre-fix).
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON optional). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_nonnumeric_offset_throws_ov_exception*. Pre-fix the test FAILS because TensorExternalData ctor throws std::invalid_argument (an uncaught non-ov exception type) instead of ov::Exception; post-fix (stoull wrapped to rethrow error::invalid_external_data) the test PASSES. A crafted external_data ONNX fixture must be added first (see TODOs).

## Suggested fix
Replace bare `std::stoull` with a helper that catches exceptions and throws an `error::invalid_external_data` or `std::runtime_error`. For example: `auto parse_uint64 = [](const std::string& s, const char* field) -> uint64_t { try { return std::stoull(s); } catch (const std::exception& e) { throw error::invalid_external_data{"Invalid " + std::string(field) + " value '" + s + "': " + e.what()}; } };` then use `m_offset = parse_uint64(entry.value(), "offset");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #447.
