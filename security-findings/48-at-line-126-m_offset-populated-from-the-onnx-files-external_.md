# Security finding #48: At line 126, `m_offset` (populated from the ONNX file's `external_d…

**Summary:** At line 126, `m_offset` (populated from the ONNX file's `external_d…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary-address read and memcpy: an attacker-controlled ONNX file on disk causes the ONNX frontend to dereference an attacker-chosen pointer and copy `m_data_length` bytes from that address into a newly allocated buffer. This enables: (a) crash/DoS if the address is unmapped; (b) sensitive information disclosure (reading arbitrary process memory — stack canaries, heap pointers, private keys, model weights, etc.) because the copied buffer is subsequently used to construct an `ov::op::v0::Constant` node accessible to the caller. Anyone who calls `ov::Core::read_model()` or `compile_model()` on an untrusted ONNX file is affected.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX file on disk (attacker-controlled) → ov::frontend::onnx ONNX-frontend parse path

## Description / Root cause
At line 126, `m_offset` (populated from the ONNX file's `external_data {key:'offset', value:'<integer>'}` entry via `std::stoull` in the TensorProto constructor at tensor_external_data.cpp:24) is cast directly to a pointer: `char* addr_ptr = reinterpret_cast<char*>(m_offset)`. At line 129, `std::memcpy` reads `m_data_length` bytes from this attacker-controlled address. The only guard (lines 121-124) only checks that both values are non-zero — it does NOT verify the tensor came from the trusted in-memory OV EP plugin path. Critically, there is no check that `m_tensor_place != nullptr` before entering the `ORT_MEM_ADDR` branch in tensor.cpp:455, so an on-disk ONNX file that sets `location=*/_ORT_MEM_ADDR_/*` in its external_data entries follows the exact same code path as a trusted in-memory initializer.

**Validator analysis:** Confirmed real and reachable. ov::Core::read_model(path) on an untrusted .onnx parses the file into a TensorProto with m_tensor_place==nullptr; has_external_data() (tensor.hpp:312) returns true when data_location==EXTERNAL; get_ov_constant() builds TensorExternalData from m_tensor_proto (tensor.cpp:449-453) reading the file-supplied 'location'/'offset'/'length' entries (tensor_external_data.cpp:20-30 via std::stoull). The dispatch at tensor.cpp:455 and the identical one at tensor.hpp:324 do NOT condition the ORT_MEM_ADDR branch on the trusted in-memory path (m_tensor_place!=nullptr), so a file can set location='*/_ORT_MEM_ADDR_/*' and follow the same path as a trusted in-memory initializer. load_external_mem_data() then reinterpret_casts m_offset to a pointer (line 126) and memcpy reads m_data_length bytes (line 129); the only check (121-124) is non-zero, with NO file_size/range bound (contrast load_external_data 83-85 and load_external_mmap_data 53-54 which DO bound offset/length). The CWE-822 Untrusted Pointer Dereference classification is accurate; impact (arbitrary-address read → DoS on unmapped address; info disclosure into the returned Constant buffer) is accurate. NOTE: the finding cites 'tensor.cpp:455' for the dispatch but the file_ref/line_ref points at tensor_external_data.cpp:126-129 — both are correct; the fix must touch the dispatch site. The proposed fix is correct and the right approach: gate the ORT_MEM_ADDR branch on m_tensor_place!=nullptr at both tensor.cpp:455 and tensor.hpp:324, and treat ORT_MEM_ADDR on the on-disk path as invalid_external_data. The additional 'reject offset below a minimum user-space address' heuristic is weak/non-portable and should be dropped — the trust-boundary gate (m_tensor_place!=nullptr) is the real, sufficient mitigation, since the whole ORT_MEM_ADDR scheme is only meaningful when the EP supplies an in-process address; there is no legitimate reason for an on-disk file to name an absolute memory address.

## Exploit / Proof of Concept
Craft an ONNX protobuf with a TensorProto whose `data_location` is set to `TensorProto_DataLocation_EXTERNAL` and whose `external_data` list contains `{key:'location', value:'*/_ORT_MEM_ADDR_/*'}`, `{key:'offset', value:'4294967296'}` (or any non-zero address of interest), and `{key:'length', value:'4096'}`. Pass this file to `ov::Core::read_model(path)`. During parsing, `has_external_data()` returns true; `TensorExternalData(*m_tensor_proto)` stores `m_offset=4294967296` and `m_data_length=4096`; `data_location() == ORT_MEM_ADDR` dispatches to `load_external_mem_data()`; the only guard passes (both non-zero); `memcpy` reads 4096 bytes from address 0x100000000 into an AlignedBuffer; the caller can inspect this buffer via the returned Constant node's data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-822 in ov::frontend::onnx.
// Pre-fix: convert_model() of a model whose initializer sets
//   external_data location='*/_ORT_MEM_ADDR_/*', offset=<arbitrary addr>, length=N
//   reaches TensorExternalData::load_external_mem_data()
//   (tensor_external_data.cpp:126 reinterpret_cast<char*>(m_offset);
//    tensor_external_data.cpp:129 std::memcpy(dst, addr_ptr, m_data_length))
//   -> arbitrary-address read (ASan: SEGV / heap-buffer-overflow on the bogus pointer).
// Post-fix: the ORT_MEM_ADDR branch at tensor.cpp:455 / tensor.hpp:324 is gated on
//   m_tensor_place != nullptr, so an on-disk file is rejected with ov::Exception
//   (error::invalid_external_data) before any dereference.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO(fixture): add a crafted model 'external_data_ort_mem_addr_ondisk.onnx' under
//   src/frontends/onnx/tests/models/ with a single initializer whose TensorProto has:
//     data_location = EXTERNAL
//     external_data = [ {key:'location', value:'*/_ORT_MEM_ADDR_/*'},
//                       {key:'offset',   value:'4096'},   // any bogus non-zero addr
//                       {key:'length',   value:'4096'} ]
//   (offset/length parsed by std::stoull at tensor_external_data.cpp:24-26).
//   convert_model() resolves model paths via the test models dir (see onnx_utils.hpp).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected_from_disk) {
    // Pre-fix this would dereference the attacker-chosen address inside
    // load_external_mem_data() before this assertion could observe a clean throw.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr_ondisk.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_rejected_from_disk*. Expected pre-fix: AddressSanitizer SEGV / 'unknown-crash' on the wild pointer inside std::memcpy at tensor_external_data.cpp:129 (test aborts instead of catching the throw). Expected post-fix: the on-disk ORT_MEM_ADDR initializer is rejected with ov::Exception (error::invalid_external_data) and the EXPECT_THROW passes. TODO: supply the crafted external_data_ort_mem_addr_ondisk.onnx fixture (a binary/protobuf model is required, so this test cannot be fully self-contained).

## Suggested fix
In `tensor.cpp` at the dispatch site (lines 455-456), gate the `ORT_MEM_ADDR` branch exclusively to the trusted in-memory path by requiring `m_tensor_place != nullptr`: `if (m_tensor_place != nullptr && ext_data.data_location() == detail::ORT_MEM_ADDR)`. For on-disk ONNX files (`m_tensor_place == nullptr`), the `ORT_MEM_ADDR` location value should be treated as an invalid/unrecognized location and throw `error::invalid_external_data`. The same guard should be applied in `tensor.hpp` at line 324 (`get_external_data`). Additionally, in `load_external_mem_data()` itself, add a range/plausibility check on `m_offset` before dereferencing (e.g., reject values below a minimum user-space address threshold), and document that this function must only be called from the trusted in-memory EP plugin path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #48.
