# Security finding #90: At line 24, `m_offset` is populated from the protobuf 'offset' fiel…

**Summary:** At line 24, `m_offset` is populated from the protobuf 'offset' fiel…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary read of process memory (sensitive data leak including weights, keys, runtime state) or, if the address is unmapped, a segfault/DoS. Exploitable by anyone who can cause the OpenVINO ONNX frontend to load a crafted model file — e.g., via ov::Core::compile_model() or the Python API — without requiring any special privilege.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Crafted ONNX model file → protobuf deserialization → TensorExternalData constructor (line 24) → load_external_mem_data() called from Tensor::get_external_data() (tensor.hpp:324–325)

## Description / Root cause
At line 24, `m_offset` is populated from the protobuf 'offset' field via `std::stoull(entry.value())` with no upper-bound or address-range validation. Inside `load_external_mem_data()`, the only guard before pointer reinterpretation is `bool is_valid_buffer = m_offset && m_data_length` (line 121) — a trivially satisfiable non-zero check. Line 126 then blindly casts the attacker-supplied integer to a pointer (`char* addr_ptr = reinterpret_cast<char*>(m_offset)`) and line 129 copies `m_data_length` bytes from that address into a new `AlignedBuffer` with no verification that the address lies within any legitimate, pre-registered shared-memory region.

**Validator analysis:** The flaw is real and reachable from openvino's trust boundary. The magic string ORT_MEM_ADDR is matched on the attacker-controllable protobuf 'location' field (data_location() reads m_data_location set at tensor_external_data.cpp:22). Because tensor.hpp:324 dispatches to load_external_mem_data() purely on that string match — and does so BEFORE any path sanitization — a crafted ONNX file with external_data location='*/_ORT_MEM_ADDR_/*', a nonzero decimal 'offset' and nonzero 'length' satisfies the trivial is_valid_buffer check (:121) and reaches the reinterpret_cast<char*>(m_offset) + std::memcpy at :126/:129. The constructor at :19-24 applies no range validation (contrast the file paths at :53-54 and :83-84 which do bound m_offset/m_data_length against file_size). vulnType CWE-822/CWE-125 is accurate (arbitrary in-process read of an attacker-chosen virtual address, or DoS/segfault on an unmapped address); impact is accurate. The proposed fix — gating tensor.hpp:324 on `m_tensor_place != nullptr` — is correct and sufficient to bar the file-deserialization path, since the ORT_MEM_ADDR pointer mechanism is only legitimate when ORT passes an in-process pointer via the (location,offset,size) constructor (built only when m_tensor_place!=nullptr). The defense-in-depth address-range registration is a good additional hardening but not strictly required once the dispatch is gated. The EP repo is na: it neither contains this code nor provides untrusted offsets.

## Exploit / Proof of Concept
Craft an ONNX model with one external-data tensor: set `external_data[key='location'] = '*/_ORT_MEM_ADDR_/*'`, `external_data[key='offset'] = '<target_addr_as_decimal>'`, `external_data[key='length'] = '<desired_read_size>'`. When the frontend's `Tensor::get_external_data()` (tensor.hpp:324) sees `data_location() == ORT_MEM_ADDR`, it calls `load_external_mem_data()`. The `m_offset && m_data_length` check passes for any nonzero pair; the memcpy at line 129 reads the specified number of bytes from the specified virtual address and returns them as tensor data. No additional check intervenes between the protobuf parse and the memcpy.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (load_external_mem_data) reached via core/tensor.hpp:324-325.
//
// What this encodes:
//   A file-deserialized ONNX tensor whose external_data 'location' equals the
//   ORT_MEM_ADDR marker ("*/_ORT_MEM_ADDR_/*") with an attacker-chosen 'offset'
//   and 'length' MUST be rejected at parse/convert time. Pre-fix the frontend
//   reinterpret_casts the offset to a pointer and memcpy's length bytes from it
//   (ASan: SEGV / heap-buffer-overflow / wild read). Post-fix (gate tensor.hpp:324
//   on m_tensor_place != nullptr) convert_model() must throw ov::Exception.
//
// NOTE: this requires a crafted model fixture; a self-contained TensorProto built
// inline is not how this test tree drives the frontend, so this is a SKELETON.

#include "onnx_utils.hpp"          // TODO: confirm include used by onnx_import.in.cpp for convert_model()
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: place crafted model under the frontend test models dir, e.g.
//   onnx/models/external_data/ort_mem_addr_wild_pointer.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data: location="*/_ORT_MEM_ADDR_/*", offset="<nonzero decimal addr>", length="4096"
static const std::string MANIFEST_DIR = ""; // TODO: set to ${ONNX_TEST_MODELS} root

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_ext_data_ort_mem_addr_rejected) {
    // TODO: replace with the convert_model helper actually used in onnx_import.in.cpp
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_wild_pointer.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_model_ext_data_ort_mem_addr_rejected*. Pre-fix expectation: ASan reports SEGV / wild read (or heap-buffer-overflow) inside std::memcpy at tensor_external_data.cpp:129 reached from load_external_mem_data via tensor.hpp:325, OR the process aborts — i.e. no ov::Exception is thrown so EXPECT_THROW fails. Post-fix expectation (gate tensor.hpp:324 on m_tensor_place!=nullptr, falling through to load_external_data which bounds offset/length): convert_model throws ov::Exception and the test passes. TODO before running: author the crafted ort_mem_addr_wild_pointer.onnx fixture and confirm the convert_model helper signature from onnx_import.in.cpp.

## Suggested fix
The `ORT_MEM_ADDR` path is intended only when ORT itself passes a trusted in-process pointer via the second constructor (`TensorExternalData(location, offset, size)`). Gate the call in `Tensor::get_external_data()` (tensor.hpp:324) on `m_tensor_place != nullptr` so file-deserialized tensors can never reach this code path: `if (m_tensor_place != nullptr && ext_data.data_location() == detail::ORT_MEM_ADDR)`. For defense-in-depth, `load_external_mem_data()` itself should also accept a set of pre-registered valid address ranges (start+length pairs supplied by the ORT integration layer) and assert that `[m_offset, m_offset+m_data_length)` lies within one of them before issuing the memcpy.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #90.
