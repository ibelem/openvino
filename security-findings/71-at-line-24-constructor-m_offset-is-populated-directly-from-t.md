# Security finding #71: At line 24 (constructor), `m_offset` is populated directly from `Te…

**Summary:** At line 24 (constructor), `m_offset` is populated directly from `Te…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker supplying a crafted ONNX model file can read an arbitrary number of bytes from any virtual address in the host process into a heap buffer, which is then materialized as a model constant (and potentially returned to the caller). This enables a full process-memory information-disclosure (bypassing ASLR, leaking secrets, cryptographic keys, etc.). If the target address is not mapped the process crashes (DoS). Affects every application that calls OpenVINO's ONNX frontend on an untrusted model file.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Serialized ONNX TensorProto loaded from an attacker-controlled disk file versus an in-process ORT-constructed TensorProto carrying a legitimately allocated shared-memory address

## Description / Root cause
At line 24 (constructor), `m_offset` is populated directly from `TensorProto.external_data['offset']` via `std::stoull(entry.value())` — a fully attacker-controlled uint64. At line 126, `load_external_mem_data()` casts it verbatim to `char*` with `reinterpret_cast<char*>(m_offset)`, then at line 129 issues `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` with no address validation. The only guard (lines 121–125) is `m_offset && m_data_length` (both non-zero), which an attacker trivially satisfies. There is no gate in `Tensor::get_ov_constant()` (tensor.cpp:449–456) that prevents a disk-sourced `TensorProto` from reaching this branch: when `m_tensor_place == nullptr`, line 453 constructs `TensorExternalData(*m_tensor_proto)` from raw protobuf fields and line 455 routes to `load_external_mem_data()` purely on the string value `"*/_ORT_MEM_ADDR_/*"`, which an attacker can embed freely in the ONNX file.

**Validator analysis:** The defect is real and reachable from OpenVINO's ONNX frontend on an untrusted model file. has_external_data() (tensor.hpp:312) returns true purely on data_location==EXTERNAL from the protobuf; with m_tensor_place==nullptr the constructor at tensor_external_data.cpp:19-30 fills m_offset/m_data_length via std::stoull on attacker strings and m_data_location with any string. When that string equals ORT_MEM_ADDR ('*/_ORT_MEM_ADDR_/*'), get_ov_constant (tensor.cpp:455) and get_external_data (tensor.hpp:324) both call load_external_mem_data(), which reinterpret_casts m_offset to a pointer and memcpy's m_data_length bytes with NO address validation — only `m_offset && m_data_length` at line 121, trivially satisfied. The vulnType CWE-822 (Untrusted Pointer Dereference) is accurate; impact (arbitrary-address read → info disclosure / DoS on unmapped pages) is accurate, though the read result is bounded by the shape/element_count consistency check at tensor.cpp:467, so the attacker must size length to match the declared shape (still fully controllable). Note element_count==0 path: if m_data_length==0 the is_empty_buffer branch returns an empty buffer harmlessly, so the dangerous path needs m_data_length>0, which the exploit uses. The proposedFix is correct in principle: the ORT_MEM_ADDR mechanism is only meaningful when the TensorProto was built in-process (m_tensor_place != nullptr), so gating load_external_mem_data() on m_tensor_place != nullptr at tensor.cpp:455 (and the matching site in tensor.hpp:324) and throwing error::invalid_external_data otherwise fully removes the disk-sourced attack surface. This is the necessary and sufficient fix. The proposed defense-in-depth lower-bound (m_offset < 65536) is weak — it only blocks null-page derefs, not arbitrary high addresses — so it should be considered cosmetic, not a substitute for the m_tensor_place gate. A cleaner alternative is to never honor a disk-parsed location string equal to ORT_MEM_ADDR at all (reject it in the TensorProto-based constructor). Both call sites (get_ov_constant AND get_external_data) must be fixed; fixing only one leaves the other exploitable.

## Exploit / Proof of Concept
Craft an ONNX protobuf: add an initializer with `data_location = EXTERNAL`; set `external_data` entries `{key:'location', value:'*/_ORT_MEM_ADDR_/*'}`, `{key:'offset', value:'<target_VA_as_decimal>'}`, `{key:'length', value:'4096'}`. When OpenVINO loads this file, `Tensor::get_ov_constant()` (tensor.cpp:453) constructs `TensorExternalData(*m_tensor_proto)` → `m_data_location == ORT_MEM_ADDR` → calls `load_external_mem_data()` → line 126 casts the decimal VA to `char*` → line 129 `memcpy` reads 4096 bytes from that address into the constant buffer, which is returned as an `ov::op::v0::Constant` containing the stolen memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   reached via openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:453-456
//
// A disk-sourced TensorProto whose external_data 'location' equals the ORT_MEM_ADDR
// marker "*/_ORT_MEM_ADDR_/*" with an attacker-chosen decimal 'offset' must NOT be
// dereferenced. Pre-fix, convert_model() reaches load_external_mem_data(), casts
// the offset to char* and memcpy's 'length' bytes from it (ASan: SEGV / heap-buffer
// read on an unmapped/arbitrary address). Post-fix, get_ov_constant() rejects the
// ORT_MEM_ADDR path when m_tensor_place == nullptr and throws ov::Exception
// (error::invalid_external_data).
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// TODO(fixture): add a crafted model file
//   onnx/models/ort_mem_addr_arbitrary_read.onnx (or generate it in-test via the
//   ONNX protobuf API) containing one initializer with:
//     data_location = EXTERNAL
//     external_data = { {"location", "*/_ORT_MEM_ADDR_/*"},
//                       {"offset", "<some decimal VA, e.g. 4096>"},
//                       {"length", "<bytes matching the declared shape, e.g. 16>"} }
//     dims/data_type chosen so element_count == shape_elements (tensor.cpp:467).
//   I cannot hand-author the binary fixture here, so the symbol name below is a
//   placeholder — confirm the exact file name once the fixture is committed.

OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_disk_rejected) {
    // Must throw rather than dereference the attacker-controlled offset.
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure OpenVINO with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_disk_rejected*. Pre-fix expected failure: AddressSanitizer SEGV / 'unknown-crash'/'SEGV on unknown address' inside std::memcpy at tensor_external_data.cpp:129 (arbitrary-address read), i.e. the EXPECT_THROW does not catch a crash. Post-fix: the test passes because Tensor::get_ov_constant() (tensor.cpp:455) throws error::invalid_external_data (an ov::Exception) for the ORT_MEM_ADDR path when m_tensor_place == nullptr. Requires the crafted .onnx fixture noted in the TODO above.

## Suggested fix
The `ORT_MEM_ADDR` mechanism is only safe when the TensorProto was constructed programmatically by ORT with a legitimate in-process pointer. Prohibit this path for disk-sourced tensors: in `Tensor::get_ov_constant()` (tensor.cpp:455), only allow `load_external_mem_data()` when `m_tensor_place != nullptr` (i.e., the TensorExternalData was built from `TensorONNXPlace` with a trusted in-process pointer). When `m_tensor_place == nullptr` and `ext_data.data_location() == ORT_MEM_ADDR`, throw `error::invalid_external_data`. Additionally, in `load_external_mem_data()` itself, add a hard lower-bound check (`if (m_offset < 65536) throw error::invalid_external_data{*this}`) as defense-in-depth to reject null-page and low-address dereferences.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #71.
