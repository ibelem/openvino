# Security finding #440: At line 121, the only guard before `reinterpret_cast<char*>(m_offse…

**Summary:** At line 121, the only guard before `reinterpret_cast<char*>(m_offse…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker who can supply a crafted ONNX model file to the OpenVINO EP can read an arbitrary range of process memory starting at any non-zero address of their choosing, for any non-zero length. This enables (a) information disclosure (leaking heap/stack secrets, ASLR-defeating pointer reads, private keys), (b) process crash/DoS when the address is unmapped (SIGSEGV/access violation), and potentially (c) exploitation of side-channel leaks. Any application that loads an untrusted ONNX model via the OV ONNX frontend or OV EP is affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file (attacker-supplied) parsed by the ov::frontend::onnx ONNX frontend → TensorExternalData constructor → load_external_mem_data

## Description / Root cause
At line 121, the only guard before `reinterpret_cast<char*>(m_offset)` (line 126) followed by `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` (line 129) is `bool is_valid_buffer = m_offset && m_data_length`, which merely checks both values are non-zero. There is no validation that `m_offset` was placed there by the legitimate ORT runtime (via the IPC/shared-memory path) rather than being attacker-supplied from a model file. The `ORT_MEM_ADDR` sentinel string check at line 117 only gates entry into the function; it does NOT distinguish model-file-derived offsets from runtime-set ones. `m_offset` is set directly from `std::stoull(entry.value())` in the TensorExternalData(const TensorProto&) constructor (line 24) with no further validation, allowing an attacker to supply any 64-bit integer as a raw pointer.

**Validator analysis:** The flaw is real and reachable in the openvino repo. TensorExternalData(const TensorProto&) (tensor_external_data.cpp:19-36) copies location, offset and length verbatim from the model file's external_data entries with no validation. In Tensor::get_ov_constant (core/tensor.cpp:448-456), when m_tensor_place is null (the standard file-load path) the proto-derived TensorExternalData is built and, because data_location()==ORT_MEM_ADDR (the string is itself stored in the file), load_external_mem_data() is invoked. There the only guard (line 121) is a non-zero test, after which line 126 casts the attacker-controlled 64-bit value to char* and line 129 memcpys m_data_length bytes from it — a classic CWE-822 untrusted-pointer-dereference / CWE-125 OOB read giving arbitrary-address read or crash. The ORT_MEM_ADDR sentinel only gates entry; it does not distinguish runtime-set addresses from file-parsed ones, exactly as the finding states. The CWE classification and impact (info disclosure + DoS) are accurate. The proposed fix is correct and sufficient in principle: gate load_external_mem_data() on a flag that is only true for the programmatic TensorExternalData(location,offset,size) constructor (the genuine ORT IPC path), throwing invalid_external_data when the object came from a TensorProto. A cleaner equivalent: in tensor.cpp only take the ORT_MEM_ADDR branch when m_tensor_place!=nullptr (i.e. constructed from runtime memory), never for the m_tensor_proto file branch. Either fully closes the file-spoofing vector. The openvinoEp repo is rejected: it holds none of the vulnerable code and the attack is reachable directly through OpenVINO's ONNX frontend; the EP conduit role is unproven from the code actually read.

## Exploit / Proof of Concept
Craft an ONNX model with an initializer tensor whose `external_data` entries are: `location = "*/_ORT_MEM_ADDR_/*"`, `offset = <target process address, e.g. 0x7fff00001000>`, `length = 4096`. When the model is loaded: (1) TensorExternalData(const TensorProto&) at line 24 stores `m_offset = stoull("0x7fff00001000")` (or any decimal representation) into the uint64_t field; (2) tensor.cpp line 453 takes the `m_tensor_proto` branch (model-file path), builds a TensorExternalData from the proto, then at line 455 sees `data_location() == ORT_MEM_ADDR` and calls `load_external_mem_data()`; (3) inside that function, `is_valid_buffer = (0x7fff00001000 != 0) && (4096 != 0) = true`, so execution falls through; (4) line 126 casts the attacker value to `char*` and line 129 memcpys 4096 bytes from that address into a new AlignedBuffer, which is then returned and used as tensor constant data — effectively leaking arbitrary process memory into model output.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-129
// A crafted ONNX model whose initializer external_data carries
//   location = "*/_ORT_MEM_ADDR_/*", offset = <arbitrary addr>, length = <n>
// reaches TensorExternalData::load_external_mem_data() via
//   core/tensor.cpp:455-456 and performs reinterpret_cast<char*>(m_offset) + memcpy.
// Pre-fix: arbitrary-address read -> ASan SEGV / heap-buffer-overflow (or silent leak).
// Post-fix: the file-parsed ORT_MEM_ADDR path must be rejected -> ov::Exception
// (error::invalid_external_data) thrown during convert_model.
//
// Style follows onnx_import.in.cpp (ov_onnx_frontend_tests).

#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // provides convert_model(...) used across onnx_import tests

using namespace ov::frontend::onnx::tests;

// TODO(fixture): add a crafted model under
//   src/frontends/onnx/tests/models/ort_mem_addr_external_data.onnx (or .prototxt)
// with an initializer tensor whose external_data entries are:
//   { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//   { key:"offset",   value:"4096" }   // any small non-zero, non-mapped offset
//   { key:"length",   value:"4096" }
// and data_location set so the model loads via the m_tensor_proto branch
// (i.e. NOT through a runtime tensor_place). The model itself must otherwise be
// a minimal valid graph (e.g. a single Identity/Add consuming the initializer).
TEST(onnx_external_data, ort_mem_addr_sentinel_from_file_is_rejected) {
    // Before the fix this convert_model dereferences attacker-controlled m_offset
    // (tensor_external_data.cpp:126) and memcpys length bytes (line 129).
    // After the fix the ORT_MEM_ADDR path must be gated to runtime-constructed
    // TensorExternalData only, so loading from file must throw.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_external_data.ort_mem_addr_sentinel_from_file_is_rejected'. Pre-fix expectation: AddressSanitizer reports SEGV on unknown address / heap-buffer-overflow inside std::memcpy at tensor_external_data.cpp:129 (or the EXPECT_THROW fails because no exception is raised while arbitrary memory is read). Post-fix expectation: convert_model throws ov::Exception (error::invalid_external_data) and the test passes. TODO: supply the crafted .onnx fixture described in the test before this can compile/run.

## Suggested fix
The `load_external_mem_data()` path must be restricted to tensors populated via the programmatic (ORT IPC) constructor `TensorExternalData(const std::string&, size_t, size_t)`, not from parsed model files. Add a boolean flag `m_from_ort_runtime` set to `true` only in the second constructor and `false` in the proto-parsing constructor. In `load_external_mem_data()`, add a guard: `if (!m_from_ort_runtime) throw error::invalid_external_data{*this};`. Alternatively, the model-loading path in `tensor.cpp::get_ov_constant` at line 455 should never call `load_external_mem_data()` when `ext_data` was constructed from a `TensorProto` (i.e., from a file). Expose a `is_runtime_mem_addr()` method (or check a flag) and skip the ORT_MEM_ADDR branch when loading from disk. This prevents a model-file-embedded sentinel from triggering the raw-pointer path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #440.
