# Security finding #298: At line 126, `m_offset` — a `uint64_t` parsed directly from the ONN…

**Summary:** At line 126, `m_offset` — a `uint64_t` parsed directly from the ONN…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read of process memory. An attacker who can supply a malicious ONNX model to any application using the OpenVINO ONNX frontend will cause the process to `memcpy` from any virtual address with any attacker-chosen length into a returned `AlignedBuffer`, giving full read access to process secrets (keys, heap contents, model weights, etc.). On 32-bit targets or with a large `m_data_length`, this can also trigger a crash/DoS. The returned buffer is consumed as tensor data, so the leaked bytes propagate to inference outputs visible to the caller.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file → TensorExternalData(const TensorProto&) constructor → get_external_data() → load_external_mem_data()

## Description / Root cause
At line 126, `m_offset` — a `uint64_t` parsed directly from the ONNX file's `external_data["offset"]` field via `std::stoull(entry.value())` in the constructor (cpp:24) — is cast to `char* addr_ptr = reinterpret_cast<char*>(m_offset)`. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads from that attacker-controlled address for attacker-controlled `m_data_length` bytes. The guard at line 117 only checks `m_data_location == ORT_MEM_ADDR`, which is also attacker-controlled (parsed from the ONNX file at cpp:22). The validity check at lines 121–124 only requires `m_offset != 0 && m_data_length > 0` — trivially satisfiable. No process address-space validation, range check, or alignment check is present.

**Validator analysis:** Confirmed real and reachable for openvino. When a model is loaded from file, m_tensor_place is nullptr so get_external_data() (tensor.hpp:318-322) builds TensorExternalData from *m_tensor_proto, whose constructor (cpp:19-30) reads location/offset/length verbatim from the file's external_data list. has_external_data() (tensor.hpp:312-313) only requires proto data_location==EXTERNAL, which is independent of the external_data 'location' string; an attacker sets that string to the ORT_MEM_ADDR sentinel '*/_ORT_MEM_ADDR_/*'. Then tensor.hpp:324 dispatches to load_external_mem_data(), whose only checks (cpp:117 location==ORT_MEM_ADDR, cpp:121-124 offset && length) are all attacker-satisfiable. cpp:126 reinterpret_cast<char*>(m_offset) + cpp:129 memcpy gives an arbitrary read of m_data_length bytes from an attacker-chosen address into the returned AlignedBuffer, which becomes tensor data — so CWE-822 'Untrusted Pointer Dereference' and the arbitrary-read/info-leak (or DoS via wild address) impact are accurate. The mmap/file paths (load_external_mmap_data, load_external_data) DO bounds-check against file_size, but the ORT_MEM_ADDR branch bypasses all of that by design — it was only ever meant for ORT-supplied in-process addresses. The proposed two-layer fix is correct and sufficient: layer (1) gating tensor.hpp:324 on m_tensor_place!=nullptr alone closes the file path; layer (2) (a from-proto flag rejected inside load_external_mem_data) is good defense-in-depth. Either layer is sufficient; both together are best. openvinoEp is na: the EP supplies weights via the legitimate in-process ORT_MEM_ADDR convention and does not expose the untrusted-file parsing path that triggers this.

## Exploit / Proof of Concept
Craft an ONNX model with a tensor whose `external_data` list contains: `{key: "location", value: "*/_ORT_MEM_ADDR_/*"}`, `{key: "offset", value: "<target_address_as_decimal>"}`, `{key: "length", value: "4096"}`. Pass the model to `ov::Core::read_model()`. During constant-tensor loading, `get_external_data()` in tensor.hpp line 322 constructs `TensorExternalData(*m_tensor_proto)`, line 324 sees `data_location() == ORT_MEM_ADDR`, and calls `load_external_mem_data()` (line 325). Inside that function, `reinterpret_cast<char*>(m_offset)` at line 126 becomes a pointer to the target address, and `memcpy` at line 129 reads 4096 bytes from it into the output buffer, leaking memory to the caller.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// A file-parsed ONNX tensor whose external_data 'location' == "*/_ORT_MEM_ADDR_/*"
// reaches load_external_mem_data(), which reinterpret_casts the file-supplied
// 'offset' to a char* and memcpy's 'length' bytes from it (arbitrary read).
// This test loads a crafted model and asserts the frontend REJECTS it instead
// of dereferencing the attacker-controlled pointer.
//
// Pre-fix: ASan reports a wild/invalid read inside std::memcpy at cpp:129
//          (or the model loads and silently leaks process memory).
// Post-fix: convert_model() throws ov::Exception (invalid_external_data),
//          because the ORT_MEM_ADDR branch is no longer honoured for
//          data parsed from m_tensor_proto.
//
// Style mirrors onnx_import.in.cpp (uses convert_model + EXPECT_THROW).

#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"  // provides convert_model(...) helper used by onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

// TODO(fixture): provide models/external_data/ort_mem_addr_arbitrary_read.onnx
//   A model with a single Constant/initializer tensor where:
//     data_location = EXTERNAL
//     external_data = [ {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                       {key:"offset",   value:"<some nonzero decimal address>"},
//                       {key:"length",   value:"4096"} ]
//   This binary fixture cannot be authored inline here; it must be added to the
//   onnx frontend test models dir (see existing external-data fixtures used by
//   onnx_import.in.cpp). Without it the test cannot run -> skeleton.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_is_rejected) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_from_file_is_rejected*. Requires the crafted fixture models/external_data/ort_mem_addr_arbitrary_read.onnx (see TODO). Expected pre-fix: AddressSanitizer SEGV / invalid read inside std::memcpy at tensor_external_data.cpp:129 (or no throw). Expected post-fix: test passes because convert_model throws ov::Exception (error::invalid_external_data).

## Suggested fix
The `ORT_MEM_ADDR` sentinel is an OnnxRuntime in-process-only convention and must never be honoured for file-parsed data. Fix in two layers: (1) In `get_external_data()` (tensor.hpp line 324), gate the `load_external_mem_data()` call on `m_tensor_place != nullptr` — i.e., only call it when the tensor was constructed from a trusted in-process `TensorONNXPlace`, not from `m_tensor_proto`: `if (m_tensor_place != nullptr && ext_data.data_location() == detail::ORT_MEM_ADDR)`. (2) Add a `bool m_from_proto` flag to `TensorExternalData`, set to `true` in the `TensorProto` constructor (cpp:19), and at line 117 add `if (m_from_proto) throw error::invalid_external_data{*this};` so `load_external_mem_data()` unconditionally rejects data originating from a parsed file.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #298.
