# Security finding #448: Line 126 reinterprets `m_offset` (a `uint64_t` parsed from the ONNX…

**Summary:** Line 126 reinterprets `m_offset` (a `uint64_t` parsed from the ONNX…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who can supply an ONNX model with `location == ORT_MEM_ADDR` (a plain string comparison — no authentication) and an arbitrary numeric `offset` and `length` can force a `memcpy` from an arbitrary virtual address. This is an out-of-bounds read that can leak heap/stack secrets or, in process-injection scenarios, crash the process (SIGSEGV/AV), constituting both an information leak and a potential RCE primitive.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model external_data entries → std::stoull-parsed `offset` field → reinterpret_cast to char* → memcpy source pointer

## Description / Root cause
Line 126 reinterprets `m_offset` (a `uint64_t` parsed from the ONNX model's `offset` external_data field via `std::stoull` at line 24 with no range validation) as a raw pointer: `char* addr_ptr = reinterpret_cast<char*>(m_offset);`. Line 129 then copies `m_data_length` bytes from this attacker-supplied address: `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length);`. The checks at lines 121–124 only validate that `m_offset` is nonzero and `m_data_length` is set; they do not restrict the pointer to a valid memory range.

**Validator analysis:** CWE-822 (Untrusted Pointer Dereference) is accurate. The TensorProto constructor (cpp:19-36) parses location/offset/length straight from the model's external_data entries with no validation, and Tensor::get_external_data (tensor.hpp:324) selects the ORT_MEM_ADDR branch using only a string comparison on that attacker-controlled value — so a file-loaded ONNX model can set location to the literal marker '*/_ORT_MEM_ADDR_/*' (note: the work item's exploit string 'ort_mem_addr' is wrong; the real constant is hpp:91) plus an arbitrary numeric offset/length, yielding reinterpret_cast<char*>(m_offset) and memcpy of m_data_length bytes from an arbitrary address (cpp:126-129). The guards at cpp:121-124 only check nonzero offset/length, not address validity. Impact (arbitrary OOB read → info leak / crash) is correct; it is read-only into a freshly allocated buffer so it is primarily a leak/DoS primitive, not a direct write-RCE. The proposed fix is correct and sufficient: the ORT_MEM_ADDR raw-pointer channel must only be reachable via the trusted in-process size_t-arg constructor (tensor.hpp:319-321 / cpp:37-41); the TensorProto file-loading constructor (cpp:19-36) should reject location==ORT_MEM_ADDR (or get_external_data should refuse the mem branch when the source is a file-parsed proto). A trust flag distinguishing the two constructors is the cleanest implementation.

## Exploit / Proof of Concept
Create an ONNX model where a tensor's `external_data` entries have: key `"location"` = `"ort_mem_addr"` (matching `ORT_MEM_ADDR`), key `"offset"` = `"4294967296"` (an interesting target address as a decimal string), and key `"length"` = `"256"`. When the ONNX frontend calls `load_external_mem_data()`, line 126 creates `addr_ptr = reinterpret_cast<char*>(0x100000000)` and line 129 reads 256 bytes from that address into `aligned_memory`. This is unmediated by any bounds check on the address value.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-822 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (reinterpret_cast<char*>(m_offset) + memcpy, reached via tensor.hpp:322-325).
// A file-loaded ONNX model whose initializer external_data sets
//   location == "*/_ORT_MEM_ADDR_/*" (the ORT_MEM_ADDR marker, hpp:91)
//   with an arbitrary numeric offset/length must be REJECTED, not dereferenced.
// Pre-fix: convert_model attempts memcpy from reinterpret_cast<char*>(offset)
//          -> ASan SEGV / heap-buffer-overflow or process crash.
// Post-fix: the file-loaded TensorProto path refuses ORT_MEM_ADDR and throws ov::Exception.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.
// NOTE: requires a crafted fixture model (see TODO) — marked skeleton accordingly.

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/ort_mem_addr_external_data.onnx (or .prototxt) whose single
//       initializer has data_location=EXTERNAL and external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"
//         offset   = "4096"   (an arbitrary, non-mappable address)
//         length   = "256"
//       Use the onnx_import.in.cpp prototxt fixture convention so the importer
//       parses it through the TensorProto ctor (tensor_external_data.cpp:19-36).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_rejects_ort_mem_addr_marker) {
    // The ORT_MEM_ADDR raw-pointer channel must be unreachable from a file-loaded model.
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
```
**Build / run:** Add the TEST to src/frontends/onnx/tests/onnx_import.in.cpp and provide the crafted fixture src/frontends/onnx/tests/models/ort_mem_addr_external_data.onnx. Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_data_rejects_ort_mem_addr_marker*'. Pre-fix expectation: AddressSanitizer SEGV / 'attempting memcpy from unaddressable region' (reinterpret_cast<char*>(offset) deref at tensor_external_data.cpp:129); post-fix expectation: ov::Exception (invalid_external_data) thrown, test passes.

## Suggested fix
The `ORT_MEM_ADDR` code path is an internal ORT EP communication channel and must never be reachable from a file-loaded ONNX model. Add a gate in the `TensorExternalData` constructor and in `extract_tensor_external_data` that rejects `ORT_MEM_ADDR` unless the model was loaded via the trusted in-process ORT path (e.g., a separate constructor overload or a boolean trust flag). At minimum, validate that `m_offset` and `m_data_length` describe a range within a pre-registered safe memory arena before performing the reinterpret_cast and memcpy.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #448.
