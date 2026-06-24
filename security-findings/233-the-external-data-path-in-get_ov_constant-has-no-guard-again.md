# Security finding #233: The external-data path in `get_ov_constant()` has no guard against …

**Summary:** The external-data path in `get_ov_constant()` has no guard against …

**CWE IDs:** CWE-843: Access of Resource Using Incompatible Type (Type Confusion)
**Severity / Impact:** When the resulting Constant node is later read (e.g., string pointer/length fields accessed), the runtime dereferences attacker-controlled bytes as `std::string` internals (pointer + size + capacity), resulting in arbitrary read/write, crash (DoS), or potentially remote code execution. Affects any application that loads an attacker-supplied ONNX model with a STRING tensor that has external data.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:462` — `Tensor::get_ov_constant()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX model and external data file read by the ONNX frontend

## Description / Root cause
The external-data path in `get_ov_constant()` has no guard against `element::string` tensors. `get_data<std::string>()` (line 414) correctly throws "External strings are not supported", but `get_ov_constant()` bypasses this by computing `element_count = constant_buffer->size() * 8 / ov_type.bitwidth()` (line 463) using the string bitwidth (8*sizeof(std::string) = 192 or 256 bits on 64-bit). For a 24-byte external file (sizeof(std::string)=24, bitwidth=192), this yields element_count=1, which satisfies a shape [1] check at line 467. `ov::op::v0::Constant(element::string, [1], constant_buffer)` is then constructed at line 480, with 24 bytes of raw attacker-controlled binary data handed to the Constant as if it were a valid serialized `std::string` object.

**Validator analysis:** CWE-843 type confusion is accurate. The external-data branch of Tensor::get_ov_constant() lacks the element::string guard that get_data<std::string>() has at tensor.cpp:414. For a STRING tensor with external data, element_count resolves to a value matching shape_size (the line 463 fallback gives file_size/sizeof(std::string), and the line 467 check passes), so execution reaches tensor.cpp:480 which calls the AlignedBuffer-taking Constant ctor (constant.cpp:301). That ctor only asserts *constant_size==data_size (constant.cpp:309) — i.e. file size == sizeof(std::string)*count — and stores the raw bytes as m_data, instead of a properly-constructed StringAlignedBuffer (cf. constant.cpp:244-246). Any later access of the string constant reinterprets those attacker-controlled bytes as std::string {ptr,size,capacity}, giving arbitrary read/write — the stated impact (DoS/potential RCE) is credible. The proposed fix (throw error::invalid_external_data for ov_type==element::string before the external-data element-count computation, mirroring line 414) is correct and sufficient; it should be placed right after has_external_data() is known true (before line 462) so both the element-count and Constant-construction paths are blocked. Reachable in openvino via convert_model on a crafted ONNX with a STRING initializer using external data. openvinoEp reachability is unproven and the flaw does not reside in EP code, hence rejected for that repo.

## Exploit / Proof of Concept
Craft an ONNX model with a 1-element tensor of type STRING (data_type=8) and a shape of [1], where `data_location` points to an external file of exactly `sizeof(std::string)` bytes (e.g., 24 bytes of arbitrary data filled with attacker-chosen pointer values). When `get_ov_constant()` is called: `ov_type = element::string`, `element_count = 0` (get_data_size returns 0 for external data without metadata), `constant_buffer` loads the 24-byte file, line 463 computes `element_count = 24*8/192 = 1`, line 467 check passes (1 == 1), and line 480 constructs `Constant(string, [1], constant_buffer)` treating those 24 bytes as a `std::string` object. Dereferencing causes memory corruption.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-843 type confusion in
//   openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:480 (Tensor::get_ov_constant)
// Unchecked path: a STRING initializer with external_data reaches the
//   Constant(element::string, shape, AlignedBuffer) ctor (constant.cpp:301) with
//   raw external-file bytes, which are later interpreted as std::string internals.
// The fix adds: if (ov_type == ov::element::string) throw error::invalid_external_data(...);
//
// This test loads a crafted ONNX model whose single initializer is type=STRING (data_type=8),
// shape=[1], with data_location pointing to an external file of exactly sizeof(std::string)
// bytes. Pre-fix: model loads (or ASan reports a heap overflow when the bytes are read as
// std::string). Post-fix: convert_model throws ov::Exception ("External string tensors are
// not supported").
//
// Harness: ov_onnx_frontend_tests (style of onnx_import.in.cpp).
//
// TODO(fixture): add the crafted model + external file under
//   onnx/frontend/tests/models/, e.g. external_data/string_external_data.onnx
//   referencing a sibling raw file of sizeof(std::string) arbitrary bytes.
//   These binary fixtures cannot be generated here.
// TODO(symbols): confirm the test helper name (FrontEndTestUtils::convert_model /
//   onnx_import test util) and the OPENVINO_TEST(${BACKEND_NAME}, ...) macro actually used
//   in this test tree before committing.

OPENVINO_TEST(${FRONTEND_NAME}_onnx, string_tensor_with_external_data_is_rejected) {
    // TODO: use the project's convert_model helper used elsewhere in onnx_import.in.cpp
    EXPECT_THROW(
        convert_model("external_data/string_external_data.onnx"),
        ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_onnx_frontend_tests --gtest_filter=*string_tensor_with_external_data_is_rejected*. Pre-fix expected: test fails — either no exception is thrown (Constant built from raw bytes) or AddressSanitizer reports a heap-buffer-overflow / SEGV when the external bytes are dereferenced as std::string {ptr,size,capacity}. Post-fix expected: convert_model throws ov::Exception (error::invalid_external_data "External string tensors are not supported") and the test passes. Requires the TODO binary fixtures (crafted .onnx + sizeof(std::string)-byte external file).

## Suggested fix
Add a type guard before the external-data element count computation: if `ov_type == ov::element::string`, throw `error::invalid_external_data("External string tensors are not supported")` (analogous to the check at line 414). Additionally, add the same guard immediately before line 462, e.g.: `if (ov_type == ov::element::string) { throw error::invalid_external_data("External string tensors are not supported for '" + get_name() + "'"); }`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #233.
