# Security finding #470: `std::stoull(entry.value())` at lines 24 and 26 is called directly …

**Summary:** `std::stoull(entry.value())` at lines 24 and 26 is called directly …

**CWE IDs:** CWE-248: Uncaught Exception
**Severity / Impact:** A crafted ONNX model with `external_data: [{key: "offset", value: "abc"}]` or `{key: "length", value: ""}` will throw an uncaught `std::invalid_argument` during model loading, crashing or aborting the hosting application. This is a reliable DoS against any inference engine or application that loads untrusted ONNX models.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24` — `TensorExternalData::TensorExternalData()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX model external_data[offset] and external_data[length] fields (attacker-controlled strings) → std::stoull

## Description / Root cause
`std::stoull(entry.value())` at lines 24 and 26 is called directly on attacker-supplied protobuf string values with no surrounding try/catch and no pre-validation. `std::stoull` throws `std::invalid_argument` if the string is not a valid integer (e.g. `"abc"`, `""`, `"-1"`) and `std::out_of_range` if the value exceeds `UINT64_MAX`. Neither exception is caught in the constructor or in its callers before reaching the application boundary.

**Validator analysis:** Confirmed real: TensorExternalData::TensorExternalData (lines 19-36) iterates tensor.external_data() and calls std::stoull(entry.value()) at lines 24 (offset) and 26 (length) on raw attacker-supplied protobuf strings with NO try/catch and NO pre-validation. std::stoull throws std::invalid_argument on non-numeric/empty input and std::out_of_range on overflow. The surrounding methods (load_external_mmap_data:46-50, load_external_data:76-80) demonstrate the codebase's own pattern of catching and converting to error::invalid_external_data — the constructor conspicuously omits this. The downstream bounds checks (lines 53-55, 65-66) only run if the constructor succeeds, so they do not mitigate. CWE-248 (Uncaught Exception) is the correct classification and DoS-on-untrusted-model-load is an accurate impact. The proposed fix (wrap stoull in try/catch converting to error::invalid_external_data, matching lines 48-50) is correct and sufficient; a digit/length pre-validation or a safe_stoull helper returning std::optional is an equally valid alternative. Note std::invalid_argument is NOT an ov::Exception, so apps that only catch ov::Exception (the documented frontend contract) will not catch it — reinforcing the uncaught-exception DoS. Unit test requires a crafted .onnx fixture with malformed external_data offset, so a compilable self-contained test isn't fully achievable here; emitting a skeleton.

## Exploit / Proof of Concept
Supply an ONNX model with `TensorProto.external_data` containing `{key: "offset", value: "not_a_number"}`. When the ONNX frontend constructs `TensorExternalData`, the constructor at line 24 calls `std::stoull("not_a_number")`, which throws `std::invalid_argument`. There is no try/catch in the constructor (lines 19-36) or in the ONNX model-loading path that converts this to a controlled error, so the exception propagates uncaught and terminates the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for CWE-248 at tensor_external_data.cpp:24/26
// (std::stoull on attacker-controlled external_data offset/length strings).
// Pre-fix: convert_model throws std::invalid_argument (uncaught by ov::Exception handlers) -> abort/DoS.
// Post-fix: ctor converts to error::invalid_external_data (an ov::Exception subclass), so EXPECT_THROW(ov::Exception) passes cleanly.

#include "onnx_utils.hpp"  // FrontEndTestUtils / convert_model helper used by onnx_import.in.cpp
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted model fixture, e.g. models/external_data/invalid_offset_nonnumeric.onnx,
//       containing a TensorProto with external_data: [{key:"location",value:"data.bin"},
//       {key:"offset",value:"not_a_number"}]. Without the fixture this test cannot run.
TEST(onnx_external_data, invalid_external_data_offset_string_is_rejected) {
    // Pre-fix this surfaces as an uncaught std::invalid_argument from std::stoull (DoS).
    // Post-fix the ctor must translate it into ov::Exception (error::invalid_external_data).
    EXPECT_THROW(convert_model("external_data/invalid_offset_nonnumeric.onnx"), ov::Exception);
}

TEST(onnx_external_data, empty_external_data_length_string_is_rejected) {
    // TODO: fixture with external_data length value == "" (empty) -> std::stoull throws std::invalid_argument.
    EXPECT_THROW(convert_model("external_data/empty_length.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_external_data.invalid_external_data_offset_string_is_rejected:onnx_external_data.empty_external_data_length_string_is_rejected'. Expected pre-fix: test FAILS because an uncaught std::invalid_argument (terminate/abort) escapes instead of ov::Exception; post-fix the constructor catches std::invalid_argument/std::out_of_range and rethrows error::invalid_external_data so EXPECT_THROW(ov::Exception) passes. Requires authoring the two .onnx fixtures noted in the TODOs.

## Suggested fix
Wrap lines 24 and 26 (and the checksum branch) in a try/catch that converts `std::invalid_argument` and `std::out_of_range` into `error::invalid_external_data`. Alternatively, validate the string before conversion (e.g., check it contains only digits and is within range) or use a helper like `ov::util::safe_stoull` that returns `std::optional<uint64_t>`. Example: `try { m_offset = std::stoull(entry.value()); } catch (const std::exception& ex) { throw error::invalid_external_data{ex.what()}; }`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #470.
