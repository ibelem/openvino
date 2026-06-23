# Security finding #93: At line 126, `m_offset` (parsed from the protobuf 'offset' field vi…

**Summary:** At line 126, `m_offset` (parsed from the protobuf 'offset' field vi…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who crafts a model with `location = "*/_ORT_MEM_ADDR_/*"`, an arbitrary `offset` value, and a non-zero `length` value can cause the process to read from any virtual address. This enables an out-of-bounds/cross-object heap or stack read, potential information disclosure of heap/stack contents, or a process crash (SIGSEGV/access violation) if the address is unmapped. On platforms where the ONNX frontend processes untrusted model files, this is a remote info-leak or crash.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model protobuf 'offset' field → TensorExternalData constructor line 24 (std::stoull) → m_offset → reinterpret_cast<char*>(m_offset) at line 126

## Description / Root cause
At line 126, `m_offset` (parsed from the protobuf 'offset' field via `std::stoull` with no range check) is directly cast to a memory pointer: `char* addr_ptr = reinterpret_cast<char*>(m_offset);`. The only guard (lines 121–125) merely checks `m_offset != 0 && m_data_length != 0`. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads `m_data_length` bytes from that attacker-controlled address. There is no verification that `addr_ptr` points to accessible memory or that `m_data_length` bytes are mapped there.

**Validator analysis:** The flaw is real and the categorisation (CWE-822 Untrusted Pointer Dereference) is accurate. The protobuf constructor (tensor_external_data.cpp:19-36) copies the 'location', 'offset' and 'length' fields verbatim from the model with no range/validity check; std::stoull only rejects non-numeric strings. get_external_data() in tensor.hpp:324 selects load_external_mem_data() purely on location==ORT_MEM_ADDR, with no check that the TensorExternalData came from the trusted in-process m_tensor_place path versus an untrusted file-based TensorProto. Inside load_external_mem_data(), the only guard is m_offset && m_data_length being non-zero, then line 126 casts the attacker integer to a pointer and line 129 memcpy's from it — an arbitrary absolute-address read whose bytes are returned to the caller (tensor.hpp:331). Impact (info-leak / crash when loading an untrusted .onnx via the OpenVINO ONNX frontend) is accurate. The proposed fix is correct in direction but option (2) (VirtualQuery/proc-maps allowlists) is overkill and fragile. The minimal sufficient fix is to make the ORT_MEM_ADDR path reachable ONLY from the trusted in-process constructor: either reject location==ORT_MEM_ADDR inside TensorExternalData(const TensorProto&) (throw error::invalid_external_data), or in get_external_data() gate `load_external_mem_data()` on `m_tensor_place != nullptr` so a file-based TensorProto can never select it. That single guard closes the boundary without OS-specific probing.

## Exploit / Proof of Concept
Craft an ONNX model protobuf with an external_data entry: location="*/_ORT_MEM_ADDR_/*", offset=<target address as integer>, length=<desired read size>. When `get_external_data()` in tensor.hpp line 324 dispatches to `load_external_mem_data()`, the guard only verifies offset != 0. Line 126 casts the attacker-chosen integer to a pointer, and line 129 copies from it into the aligned buffer. The returned buffer's data is subsequently consumed by `get_external_data()` at tensor.hpp line 331, potentially exposing its contents to the caller.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (load_external_mem_data reinterpret_casts the model-controlled offset to a
//  pointer and memcpy's m_data_length bytes from it).
//
// What this encodes: a file-based ONNX model whose external_data uses
//   location = "*/_ORT_MEM_ADDR_/*", offset = <arbitrary integer>, length = <n>
// must be REJECTED by the frontend instead of dereferencing the integer as a
// pointer. Pre-fix: under ASan this aborts with a SEGV / heap-buffer-overflow
// (or wild read) in load_external_mem_data. Post-fix: convert_model throws
// ov::Exception (error::invalid_external_data) before any memcpy.
//
// Harness: ov_onnx_frontend_tests, style of onnx_import.in.cpp.
//
// SKELETON: building the crafted fixture by hand is required because
// ORT_MEM_ADDR normally only originates from the trusted in-process path, never
// from a serialized model on disk, so no existing test .onnx exercises it.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted model file under the frontend test models dir, e.g.
//   onnx/external_data/ort_mem_addr_arbitrary_offset.onnx
// whose single initializer has data_location=EXTERNAL and external_data:
//   key="location" value="*/_ORT_MEM_ADDR_/*"
//   key="offset"   value="0xdeadbeef"   (any non-zero integer)
//   key="length"   value="64"
// (Author with onnx.helper / protobuf text; commit as a binary fixture.)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_rejected) {
    // TODO: confirm exact helper name (convert_model vs. import_onnx_model)
    //       from onnx_import.in.cpp in this checkout.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_offset.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_rejected*. Expected pre-fix (ASan build): the process aborts inside detail::TensorExternalData::load_external_mem_data with a SEGV/'unknown-crash' or heap-buffer-overflow on the std::memcpy at tensor_external_data.cpp:129 (wild read from reinterpret_cast<char*>(m_offset)). Expected post-fix: the test passes because convert_model throws ov::Exception (error::invalid_external_data) before the memcpy. NOTE: requires authoring the crafted .onnx fixture described in the TODO.

## Suggested fix
The `load_external_mem_data()` path is designed for ORT shared-memory interop where the address is supplied by a trusted in-process caller. The fix must enforce that this path is never reachable from an untrusted (file-based) ONNX model: (1) Reject the ORT_MEM_ADDR path when the TensorExternalData object was constructed from a TensorProto (i.e., from a file) rather than from a trusted in-process source. Add a constructor flag or a separate code path. (2) If shared-memory use is intentional from a model file, add an allowlist/whitelist of permitted address ranges, or validate via OS APIs (e.g., VirtualQuery on Windows, /proc/self/maps on Linux) that [addr_ptr, addr_ptr+m_data_length) is a valid mapped readable region before the memcpy.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #93.
