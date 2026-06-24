# Security finding #258: At line 463, `element_count = constant_buffer->size() * 8 / ov_type…

**Summary:** At line 463, `element_count = constant_buffer->size() * 8 / ov_type…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Silent wrong-value injection: a scalar model weight supplied via an external data file is silently replaced by a zero constant. An adversary who controls the external weights file (or causes a partial/truncated write) can cause a machine-learning model to use a wrong weight value without any parse error being raised, potentially subverting inference results undetectably. Any downstream consumer of `get_ov_constant()` that relies on the returned Constant reflecting the file content is affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:462` — `Tensor::get_ov_constant()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model + external weights files supplied by an external user

## Description / Root cause
At line 463, `element_count = constant_buffer->size() * 8 / ov_type.bitwidth()` performs integer division that truncates to 0 whenever the external file contains fewer bytes than one element requires (e.g., 7 bytes for INT64 which needs 8). The mismatch-detection guard at line 467 — `!(element_count == 0 && m_shape.empty())` — then explicitly bypasses the error throw because `element_count` is 0 and the tensor is declared as a scalar (`m_shape.empty()`). Control falls to line 476 and `make_failsafe_constant(ov_type)` is returned silently, discarding the loaded external-file content with no exception, no log, and no diagnostic to the caller.

**Validator analysis:** The flaw is real and reachable from the ONNX frontend trust boundary via convert_model on a crafted model. get_data_size() (tensor.hpp:362-373) returns ext_data.size()/itemsize using integer division, so a declared external-data length of 1..(itemsize-1) bytes yields element_count==0 before loading. The buffer actually loaded is non-empty, but the recompute at tensor.cpp:463 (`size()*8/bitwidth`) again truncates to 0, and the scalar exception clause at line 467 then bypasses the mismatch throw, so common::make_failsafe_constant returns a zero-valued scalar instead of rejecting the malformed file. CWE-20 is the correct categorization; the 'silent wrong-value injection' impact is accurate but the severity is correctness/integrity (no memory-safety/crash). The proposed fix is correct in direction: the scalar suppression clause must additionally require that no external buffer was loaded (`!constant_buffer`), or equivalently validate the recomputed element_count against shape_elements whenever constant_buffer is non-null. A tighter form: change line 467's clause to `!(element_count == 0 && m_shape.empty() && !constant_buffer)` so a loaded-but-undersized external file always throws while genuinely-empty scalars still pass. Triggering requires a crafted .onnx + truncated external weights file, so a self-contained compilable test needs binary fixtures (skeleton below).

## Exploit / Proof of Concept
Craft an ONNX model with a scalar initializer (dims: empty or dims: {0}) that references an external data file and mark it EXTERNAL. Supply an external file containing fewer bytes than `ov_type.bitwidth() / 8` (e.g., 7 bytes for FLOAT64 / INT64). `get_data_size()` returns 0 for external data before loading (line 437), so `element_count` starts at 0. After loading, `constant_buffer` is non-null and non-empty (7 bytes). Line 462 condition (`element_count == 0 && constant_buffer`) is true. Line 463 computes `7*8/64 = 0` — `element_count` remains 0. Line 467 guard: `(0 != 1)` is true but `!(0==0 && true)` = false so the combined condition is false — no exception. Line 476: `element_count == 0` → `make_failsafe_constant` returned. The 7 loaded bytes are silently discarded.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for tensor.cpp:462-467 (Tensor::get_ov_constant).
// Encodes the fix: an INT64 *scalar* initializer (shape: []) whose EXTERNAL
// data file/declared length is shorter than one element (e.g. 7 bytes for INT64)
// must be REJECTED with ov::Exception, instead of being silently replaced by a
// zero failsafe constant. Pre-fix this convert_model() succeeds and yields a 0
// scalar; post-fix it throws error::invalid_external_data.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
//
// NOTE: this needs binary fixtures that the source tree does not yet contain:
//   - models/scalar_int64_truncated_external_data.onnx  (scalar initializer,
//     data_location=EXTERNAL, external_data 'length'=7)
//   - the referenced raw weights file containing only 7 bytes
// Hence this is a SKELETON; create the fixtures before enabling.

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scalar_truncated_external_data_throws) {
    // TODO: generate fixture 'scalar_int64_truncated_external_data.onnx' with a
    //       scalar (dims empty) INT64 initializer referencing an external file
    //       whose declared length is 7 (< 8 = sizeof(int64)). The companion raw
    //       data file must contain exactly 7 bytes.
    EXPECT_THROW(convert_model("scalar_int64_truncated_external_data.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_model_scalar_truncated_external_data_throws*. Pre-fix the test FAILS (convert_model returns a 0-valued failsafe scalar, no throw); after tightening the line-467 guard to require !constant_buffer, convert_model throws error::invalid_external_data and the EXPECT_THROW passes. Fixtures (crafted .onnx + 7-byte external weights file) must be authored under the onnx frontend test models dir first.

## Suggested fix
After the recomputation at line 463, add an explicit check that validates the recomputed `element_count` matches `shape_size(m_shape)` (which is 1 for a scalar). For example, after line 464, add: `if (constant_buffer && element_count != shape_elements) { throw error::invalid_external_data("The size of the external data file does not match the byte size of an initializer '" + get_name() + "' in the model"); }` — and remove or tighten the scalar exception clause in the existing guard at line 467 so it only suppresses errors when `constant_buffer` is null (i.e., there truly is no data), not when the file was loaded but the size doesn't match.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #258.
