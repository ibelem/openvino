# Security finding #411: At line 126, `m_offset` is blindly reinterpreted as a raw `char*` p…

**Summary:** At line 126, `m_offset` is blindly reinterpreted as a raw `char*` p…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary memory read (and potential process crash/SIGSEGV) when loading a malicious `.onnx` model. An attacker who can supply the model file can read `m_data_length` bytes from any virtual address they choose. Depending on address layout, this can leak heap/stack secrets, cryptographic keys, or process memory into the loaded tensor constant, which may subsequently be serialized to disk or transmitted over a network. On unmapped addresses it causes an immediate crash (DoS). All users of the OpenVINO ONNX frontend that load untrusted `.onnx` files are affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX protobuf `external_data` field (on-disk attacker-controlled model) vs. in-process pointer assumed to originate from a trusted ORT runtime

## Description / Root cause
At line 126, `m_offset` is blindly reinterpreted as a raw `char*` pointer (`reinterpret_cast<char*>(m_offset)`) and then used as the source address for `std::memcpy` at line 129 with no validation that the address falls within any legitimate mapped region. The only guard (lines 121–124) rejects the case `m_offset==0 && m_data_length!=0`, but any non-zero `m_offset` together with any non-zero `m_data_length` passes. `m_offset` is populated at construction time (line 24, `m_offset = std::stoull(entry.value())`) directly from the protobuf `external_data[offset]` string, which is attacker-controlled. The `ORT_MEM_ADDR` sentinel check (line 117) only verifies that `m_data_location` string equals `"*/_ORT_MEM_ADDR_/*"`—a string an attacker can trivially embed in an on-disk `.onnx` file.

**Validator analysis:** Confirmed reachable in openvino: the protobuf constructor (tensor_external_data.cpp:19-36) stores the attacker-controlled `location` string verbatim into m_data_location and `offset` into m_offset via std::stoull. tensor.hpp's get_external_data() (lines 318-325) selects the ORT_MEM_ADDR branch purely on the string value `data_location()==ORT_MEM_ADDR`, with no check that the object originated from the trusted runtime-pointer overload (line 37) rather than from on-disk protobuf. load_external_mem_data() only validates m_data_location==ORT_MEM_ADDR (line 117) and that offset/length are non-zero (lines 121-124); it never validates that m_offset is a legitimate mapped address before reinterpret_cast<char*> (126) and std::memcpy (129). This is a genuine CWE-822/CWE-125: arbitrary-address read of m_data_length bytes into a tensor constant, or SIGSEGV/DoS on unmapped pages. The vulnType and impact are accurate. The proposed fix is correct and sufficient: the cleanest variant is to reject ORT_MEM_ADDR as a `location` value during protobuf parsing (in the TensorProto constructor at lines 21-22, throw error::invalid_external_data when entry.value()==ORT_MEM_ADDR), since the ORT_MEM_ADDR raw-pointer path is only ever legitimate via the in-process TensorExternalData(location,offset,size) overload used when m_tensor_place!=nullptr. The m_from_protobuf-flag alternative is equally valid. Note: only the protobuf-parse path (m_tensor_place==nullptr) is affected; the runtime-pointer path remains valid. For the EP repo the code is absent and the marker is produced by trusted ORT, so it is na.

## Exploit / Proof of Concept
Craft an `.onnx` file with a tensor whose `data_location == TensorProto_DataLocation_EXTERNAL` and `external_data` entries: `key="location" value="*/_ORT_MEM_ADDR_/*"`, `key="offset" value="<target_virtual_address_as_decimal>"`, `key="length" value="4096"`. When this file is opened via `ov::Core::read_model(path)`, the `Tensor` object is constructed from the protobuf via `Tensor(TensorProto&, model_dir, mmap_cache)`. `get_external_data()` at `tensor.hpp:324` checks `data_location() == ORT_MEM_ADDR` — true — and calls `load_external_mem_data()`. `m_offset` carries the attacker's integer (set by `std::stoull` at `tensor_external_data.cpp:24`). The guard at lines 121–124 passes (both values non-zero). Line 126 casts it to `char*`; line 129 copies 4096 bytes from the target address into an `AlignedBuffer`, making the stolen bytes available as tensor constant data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data).
//
// Pre-fix: an on-disk .onnx whose tensor has data_location=EXTERNAL with
// external_data entries location="*/_ORT_MEM_ADDR_/*", offset=<attacker addr>,
// length=N reaches tensor.hpp:325 -> load_external_mem_data(), which
// reinterpret_casts the protobuf-supplied offset to char* and memcpy's N bytes
// from an arbitrary address (ASan: SEGV / heap-buffer-overflow read, or silent
// arbitrary read).
// Post-fix: the protobuf parse path rejects ORT_MEM_ADDR (or marks the object as
// protobuf-sourced) and convert_model throws ov::Exception instead.
//
// This test follows the onnx_import.in.cpp style: load a crafted model and
// assert it is rejected. It REQUIRES a crafted fixture, so it is a skeleton.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers used by ov_onnx_frontend_tests

using namespace ov::frontend::onnx::tests;

// TODO: add a crafted fixture, e.g.
//   models/external_data/ort_mem_addr_arbitrary_read.onnx (or .prototxt) with:
//     graph.initializer[0].data_location = EXTERNAL
//     external_data { key:"location" value:"*/_ORT_MEM_ADDR_/*" }
//     external_data { key:"offset"   value:"4096" }   // bogus raw address
//     external_data { key:"length"   value:"4096" }
// matching the naming/loader convention you observe in the existing
// onnx_import_external_data.in.cpp tests (read that file for the exact helper).
TEST(onnx_external_data, DISABLED_ort_mem_addr_marker_rejected_from_protobuf) {
    // TODO: confirm the exact convert_model overload + relative-path helper used
    // by onnx_import_external_data.in.cpp in this tree.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON for ASan). Add the crafted fixture under the frontend's onnx models dir used by onnx_import_external_data.in.cpp, enable the test (remove DISABLED_ and fix the convert_model path helper), then run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.*ort_mem_addr*. Pre-fix expectation: ASan reports SEGV / out-of-bounds read inside TensorExternalData::load_external_mem_data (memcpy from reinterpret_cast<char*>(m_offset)). Post-fix expectation: convert_model throws ov::Exception (error::invalid_external_data) and the test passes with no ASan report.

## Suggested fix
The `ORT_MEM_ADDR` path is only valid when `TensorExternalData` is constructed via the runtime-pointer overload (`TensorExternalData(location, offset, size)` at line 37), not from a protobuf. Add a boolean member `m_from_protobuf` (default `false`, set to `true` only in the `TensorProto` constructor at line 19). In `load_external_mem_data()`, reject the call if `m_from_protobuf` is true: `if (m_from_protobuf) throw error::invalid_external_data{*this};`. Alternatively, refuse to accept `ORT_MEM_ADDR` as a valid `location` value during protobuf parsing (line 22): add `if (entry.value() == ORT_MEM_ADDR) throw error::invalid_external_data{...};` immediately after setting `m_data_location`. Either change ensures on-disk ONNX data can never reach the raw-pointer code path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #411.
