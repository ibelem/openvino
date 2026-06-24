# Security finding #212: Lines 24 and 26 call `std::stoull(entry.value())` on the attacker-c…

**Summary:** Lines 24 and 26 call `std::stoull(entry.value())` on the attacker-c…

**CWE IDs:** CWE-248: Uncaught Exception
**Severity / Impact:** Unhandled C++ exception propagating out of model parsing, causing process abort or unwind through caller frames that do not expect it. Immediate crash/DoS for any application that loads an untrusted ONNX model with malformed external data offset or length fields.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** ONNX protobuf `external_data` key-value pairs (attacker-supplied) → `std::stoull` with no surrounding try/catch

## Description / Root cause
Lines 24 and 26 call `std::stoull(entry.value())` on the attacker-controlled `offset` and `length` string values with no try/catch. `std::stoull` throws `std::invalid_argument` if the string is not a valid integer and `std::out_of_range` if the value exceeds `ULLONG_MAX`. Neither exception is caught in this constructor, nor is there any pre-validation of the string content before conversion.

**Validator analysis:** The flaw is real: lines 24 and 26 of tensor_external_data.cpp invoke std::stoull(entry.value()) directly on the protobuf external_data string values with no validation and no surrounding try/catch, unlike every other error path in the same file which throws error::invalid_external_data (see lines 48-49,55,66 and exceptions.hpp:38-42). It is reachable from the ONNX frontend trust boundary: when a TensorProto has data_location EXTERNAL (tensor.hpp:312-313), get_external_data() constructs TensorExternalData(*m_tensor_proto) (tensor.hpp:322), running the constructor on attacker data. A value like 'offset=abc' throws std::invalid_argument; a value > ULLONG_MAX throws std::out_of_range. CWE-248 is the correct category. The 'process abort' impact is overstated — std::invalid_argument/out_of_range both derive from std::exception, so a top-level catch(std::exception) at the API boundary would catch it; the real defect is an inconsistent, poorly-typed error leaking out of model parsing rather than a clean invalid_external_data, which can still unwind through non-exception-safe caller frames. The proposed fix (wrap both stoull calls in try/catch and rethrow as error::invalid_external_data) is correct and sufficient; it matches the file's existing error conventions and makes the exception type predictable. Suggest also guarding the m_offset/m_data_length against semantic overflow, but the catch is the essential fix.

## Exploit / Proof of Concept
Supply an ONNX model with an external tensor whose `external_data` protobuf contains `key="offset", value="not_a_number"` or `value="99999999999999999999999999"` (> ULLONG_MAX). When `TensorExternalData` is constructed, `std::stoull("not_a_number")` throws `std::invalid_argument`; if this propagates uncaught, the application terminates.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:24,26 (std::stoull on attacker-controlled
// external_data offset/length). Pre-fix: a non-numeric/overflowing offset value escapes as
// std::invalid_argument / std::out_of_range (NOT an ov::Exception), so EXPECT_THROW(...,
// ov::Exception) fails (uncaught logic_error). Post-fix: the constructor converts it to
// error::invalid_external_data (an ov::Exception) and the assertion passes.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: a compilable test requires a crafted .onnx model whose tensor uses
//       data_location=EXTERNAL with external_data {key:"offset", value:"not_a_number"}.
//       That binary fixture is not authored here, hence this is a SKELETON.
#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers used by onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

// TODO: create models/external_data/bad_offset_not_a_number.onnx — a model with one
//       initializer tensor whose data_location=EXTERNAL and external_data contains
//       {key:"offset", value:"not_a_number"} (or value:"99999999999999999999999999").
TEST(onnx_external_data, malformed_offset_throws_ov_exception) {
    // Pre-fix this aborts/escapes via std::invalid_argument; post-fix it is invalid_external_data.
    EXPECT_THROW(convert_model("external_data/bad_offset_not_a_number.onnx"), ov::Exception);
}

// TODO: create models/external_data/bad_length_overflow.onnx with external_data
//       {key:"length", value:"99999999999999999999999999"} (> ULLONG_MAX).
TEST(onnx_external_data, overflow_length_throws_ov_exception) {
    EXPECT_THROW(convert_model("external_data/bad_length_overflow.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='onnx_external_data.*'. Provide the two crafted .onnx fixtures (TODO above) under the frontend test models dir. Pre-fix expectation: test fails because std::stoull throws std::invalid_argument/std::out_of_range (a std::logic_error, not ov::Exception) — under ASan/EXPECT_THROW this surfaces as an unexpected exception type / uncaught exception. Post-fix: constructor rethrows error::invalid_external_data (ov::Exception) and both tests pass.

## Suggested fix
Wrap both `std::stoull` calls in a try/catch block and convert any caught `std::invalid_argument` / `std::out_of_range` into an `error::invalid_external_data` (consistent with how other errors in this file are reported): `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& e) { throw error::invalid_external_data{"invalid offset: " + std::string(e.what())}; }`. Apply the same pattern to the `length` field.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #212.
