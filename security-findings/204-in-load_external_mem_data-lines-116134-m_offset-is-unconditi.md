# Security finding #204: In `load_external_mem_data()` (lines 116–134), `m_offset` is uncond…

**Summary:** In `load_external_mem_data()` (lines 116–134), `m_offset` is uncond…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary read-what-where primitive within the process address space. An attacker who can supply a crafted ONNX model (e.g., via a model-serving endpoint, file upload, or API accepting raw ONNX bytes) can cause the process to memcpy from any virtual address it supplies as `offset` into a freshly-allocated AlignedBuffer, then embed that data into a constant tensor node. This constitutes a full process memory disclosure (stack, heap, mapped libraries, secrets). If the target address is unmapped, the process crashes (DoS). On 64-bit Linux/Windows the full 64-bit address space is addressable as a target.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file/bytes → OV ONNX frontend protobuf parser → TensorExternalData constructor → load_external_mem_data()

## Description / Root cause
In `load_external_mem_data()` (lines 116–134), `m_offset` is unconditionally cast to a pointer (`char* addr_ptr = reinterpret_cast<char*>(m_offset)`) and used as the source for `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)`. The value of `m_offset` is populated directly from the ONNX protobuf `external_data` field via `std::stoull(entry.value())` in the `TensorExternalData(const TensorProto&)` constructor (tensor_external_data.cpp:24), with no restriction preventing a plain disk/buffer ONNX file from setting `location = "*/_ORT_MEM_ADDR_/*"`. The dispatch in `tensor.cpp:455` routes to this path purely by string equality — there is no caller-identity check or session-scope guard that restricts the ORT_MEM_ADDR branch to an in-process ORT session. The only guard (`bool is_valid_buffer = m_offset && m_data_length`, line 121) is trivially satisfied by any non-zero attacker-supplied values.

**Validator analysis:** Confirmed real and reachable for openvino. The ONNX frontend constructor TensorExternalData(const TensorProto&) (tensor_external_data.cpp:19-36) stores the `location`, `offset`, `length` strings verbatim from the protobuf with no validation; `offset` becomes a raw uint64 via std::stoull (line 24). In Tensor::get_ov_constant() (tensor.cpp:447-456) a file/buffer-loaded model has m_tensor_place == nullptr, so the external-data object is built from the untrusted proto, and the dispatch at line 455 selects load_external_mem_data() purely on string equality with the public constant ORT_MEM_ADDR = "*/_ORT_MEM_ADDR_/*" (tensor_external_data.hpp:91). load_external_mem_data() (lines 116-134) only guards `m_offset && m_data_length` (line 121) then reinterpret_casts m_offset to char* and memcpys m_data_length bytes (lines 126-129). Unlike load_external_mmap_data/load_external_data (which bound offset+length against the real file size at lines 53-54, 83-84), the ORT_MEM_ADDR path performs no bounds or provenance check. The vuln type (CWE-822 untrusted-pointer-dereference / CWE-125 OOB read) and impact (arbitrary read-what-where → memory disclosure, or DoS on unmapped address) are accurate. The proposed fix is on the right track but the simplest correct fix is to gate the dispatch on provenance: at tensor.cpp:455 require `m_tensor_place != nullptr` (the genuine in-process ORT path that supplies a real data pointer) before routing to load_external_mem_data(); equivalently add an explicit `m_from_ort_session` flag set only by the TensorONNXPlace path. Defense-in-depth: load_external_mem_data() should refuse to run when constructed from the bare TensorProto path. Validating only m_offset against an allowed-range list is weaker and harder to get right than gating on provenance.

## Exploit / Proof of Concept
Craft a minimal ONNX protobuf with one initializer tensor whose `external_data` list contains: `{key: "location", value: "*/_ORT_MEM_ADDR_/*"}`, `{key: "offset", value: "<decimal address of target memory, e.g. a stack cookie or libc .got entry>"}`, `{key: "length", value: "8"}`. Feed this file to any OpenVINO API that loads ONNX models (e.g., `ov::Core::read_model`). In `Tensor::get_ov_constant()` (tensor.cpp:453) `m_tensor_place` is null for a file-loaded model so `TensorExternalData(*m_tensor_proto)` is called; the constructor stores the attacker-controlled strings verbatim; the string-equality check at line 455 passes; `load_external_mem_data()` casts `m_offset` to `char*` and memcpys `m_data_length` bytes from it; the resulting constant tensor exposes those bytes via normal graph evaluation or serialization.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for tensor_external_data.cpp:126-129 (CWE-822/CWE-125).
// A disk/buffer-loaded ONNX model whose initializer carries external_data
// location = "*/_ORT_MEM_ADDR_/*" with an attacker-chosen `offset` must NOT be
// dereferenced as a raw pointer. Pre-fix: tensor.cpp:455 routes to
// load_external_mem_data() purely on string equality and memcpys from the
// attacker address (ASan: heap/unknown-address read or SEGV). Post-fix: the
// ORT_MEM_ADDR branch is gated on in-process provenance (m_tensor_place != null),
// so this proto-only model is rejected with ov::Exception.
//
// TODO(fixture): this test needs a crafted model `ort_mem_addr_external_data.onnx`
//   with a single FLOAT initializer "x" of shape {2} whose TensorProto.external_data =
//   { (location, "*/_ORT_MEM_ADDR_/*"), (offset, "4096"), (length, "8") }.
//   Generate it with onnx.helper (set raw_data empty, data_location=EXTERNAL) and
//   drop it under the onnx frontend test models dir. Replace the path below.
// TODO(symbols): confirm the convert_model helper / models path macro used by the
//   surrounding onnx_import.in.cpp suite (e.g. util::path_join, ONNX_TEST_MODELS).

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // convert_model(...)

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer, reject_ort_mem_addr_external_data_from_file) {
    // Must throw rather than memcpy from the attacker-supplied raw address.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_importer.reject_ort_mem_addr_external_data_from_file . Pre-fix expectation: AddressSanitizer reports a read on an unknown/invalid address (or SEGV) inside TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129 memcpy). Post-fix: the model is rejected and the EXPECT_THROW(ov::Exception) passes with no sanitizer report. Requires the crafted ort_mem_addr_external_data.onnx fixture noted in the test TODO.

## Suggested fix
Restrict the ORT_MEM_ADDR branch to models that originated from an ORT in-memory session — not from disk or a byte buffer. One approach: add a boolean flag `m_from_ort_session` to the `Tensor` class that is only set by the ORT-specific construction path (TensorONNXPlace with a valid in-process data pointer), and gate the `ORT_MEM_ADDR` dispatch on that flag (`if (ext_data.data_location() == detail::ORT_MEM_ADDR && m_from_ort_session)`). Alternatively, validate that `m_offset` falls within a known safe memory range (e.g., an externally registered allowed-range list) before dereferencing. As a defense-in-depth measure, `load_external_mem_data()` should also be changed to throw `error::invalid_external_data` unconditionally when called from the `TensorExternalData(const TensorProto&)` constructor path, since that path always represents untrusted file input.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #204.
