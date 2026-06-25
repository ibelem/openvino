# Security finding #419: At line 126, `m_offset` (a `uint64_t` read verbatim from an attacke…

**Summary:** At line 126, `m_offset` (a `uint64_t` read verbatim from an attacke…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker who supplies a crafted ONNX model can read an arbitrary span of the loading process's virtual memory into a tensor buffer (info leak: stack/heap secrets, ASLR bases, cryptographic keys). If the chosen address is unmapped the process crashes (DoS). Affects any application that calls `ov::Core::read_model()` on an attacker-supplied ONNX file with OpenVINO's ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** TensorProto.external_data field in an untrusted ONNX model file → TensorExternalData constructor (line 22/24) → m_offset/m_data_length → load_external_mem_data()

## Description / Root cause
At line 126, `m_offset` (a `uint64_t` read verbatim from an attacker-supplied protobuf string via `std::stoull` at constructor line 24) is reinterpreted as a raw pointer: `char* addr_ptr = reinterpret_cast<char*>(m_offset)`. Line 129 then performs `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` where both the source address and the byte count are fully attacker-controlled. The only guard before this path (line 117–124) checks (a) that `m_data_location == ORT_MEM_ADDR` — meaning it _allows_ the path when the attacker sets the location to `"*/_ORT_MEM_ADDR_/*"` — and (b) that `m_offset != 0 || m_data_length == 0`, which any non-zero address trivially satisfies. There is no allowlist or runtime check that prevents a file-loaded ONNX model from carrying the ORT_MEM_ADDR marker; the `get_external_data()` call in tensor.hpp:324 simply compares the string and dispatches to `load_external_mem_data()` without any origin check.

**Validator analysis:** Confirmed real and reachable in OpenVINO. The proto constructor (tensor_external_data.cpp:19-36) does not distinguish file-sourced external_data from the in-process (string,offset,size) constructor; it stores m_data_location, m_offset, m_data_length verbatim from an attacker-controlled protobuf. has_external_data() only requires data_location==EXTERNAL; get_external_data() at tensor.hpp:324 routes ANY tensor whose external_data 'location' string equals "*/_ORT_MEM_ADDR_/*" to load_external_mem_data() BEFORE any path sanitization. The guard at line 121-124 only rejects offset==0/length>0 mismatch; a non-zero offset + non-zero length passes, then line 126 casts m_offset to char* and line 129 memcpys m_data_length bytes from that arbitrary address into the returned constant. No allowlist, origin flag, or bounds check exists. This is reachable from ov::Core::read_model() on an untrusted .onnx, so CWE-822 (untrusted pointer deref) / CWE-125 (OOB read) and the arbitrary-read/DoS impact are accurate. The proposed fix is correct and sufficient: simplest robust form is to reject ORT_MEM_ADDR inside the TensorProto constructor (throw error::invalid_external_data when m_data_location==ORT_MEM_ADDR parsed from a proto), so file-sourced models can never reach the pointer-reinterpret path; the private m_from_mem_addr flag set only by the (string,offset,size) constructor is an equally valid, more surgical alternative. Marked openvinoEp na because the defect neither lives in nor requires the EP plugin_impl code to trigger.

## Exploit / Proof of Concept
Craft an ONNX model with one initializer whose `external_data` contains: `{key="location", value="*/_ORT_MEM_ADDR_/*"}`, `{key="offset", value="<decimal target address, e.g. stack or heap address>"}`, `{key="length", value="4096"}`. On model load: (1) `TensorExternalData(TensorProto&)` stores the strings into `m_data_location`, `m_offset=<target>`, `m_data_length=4096` with no validation. (2) `get_external_data()` in tensor.hpp:324 sees `ORT_MEM_ADDR` and calls `load_external_mem_data()`. (3) The sole guard at line 121–124 passes because `m_offset != 0 && m_data_length != 0`. (4) Line 126 casts the attacker integer to a `char*`; line 129 memcpy's 4096 bytes from that address into `aligned_memory`. The resulting buffer is returned as a tensor constant and may subsequently be serialised/logged or compared, leaking the captured memory to the attacker.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data) reached via
//   core/tensor.hpp:324 -> get_external_data() for a file-sourced TensorProto
//   whose external_data 'location' == "*/_ORT_MEM_ADDR_/*".
//
// Pre-fix: convert_model() on the crafted model reaches
//   reinterpret_cast<char*>(m_offset) + memcpy(...) -> arbitrary-address read
//   (ASan: SEGV / heap-buffer-overflow / use-after-poison on the wild address).
// Post-fix: the ORT_MEM_ADDR location parsed from a deserialized TensorProto is
//   rejected, so convert_model throws ov::Exception and no wild read occurs.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// NOTE: this needs a crafted binary .onnx fixture; emitted as a SKELETON.

#include "gtest/gtest.h"
#include "common_test_utils/test_constants.hpp"
#include "onnx_utils.hpp"  // for convert_model / FrontEndTestUtils helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, reject_ort_mem_addr_location_from_file) {
    // TODO: add fixture models/external_data/ort_mem_addr_marker.onnx with ONE
    //       initializer carrying data_location=EXTERNAL and external_data entries:
    //         {key="location", value="*/_ORT_MEM_ADDR_/*"}
    //         {key="offset",   value="<some non-zero address, e.g. 0xdeadbeef>"}
    //         {key="length",   value="4096"}
    //       (Cannot author a binary protobuf fixture in a read-only tree.)
    //
    // EXPECT_THROW encodes the fix: a file-sourced ORT_MEM_ADDR marker must be
    // rejected instead of being reinterpreted as a raw pointer.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_marker.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.reject_ort_mem_addr_location_from_file . Pre-fix expected: ASan SEGV / wild-address read (or heap overflow) inside TensorExternalData::load_external_mem_data at tensor_external_data.cpp:129 (memcpy from reinterpret_cast<char*>(m_offset)). Post-fix expected: convert_model throws ov::Exception (invalid_external_data), test passes. TODO: supply the crafted models/external_data/ort_mem_addr_marker.onnx fixture before the test is runnable.

## Suggested fix
The `ORT_MEM_ADDR` path is an ORT-internal IPC mechanism, not a valid path for externally loaded model files. The correct fix is to gate `load_external_mem_data()` (and the dispatch in `get_external_data()`) so that it is only reachable when the `TensorExternalData` object was constructed via the internal `(location, offset, size)` constructor — i.e., from in-process ORT shared-memory registration — never from a deserialized `TensorProto`. One clean approach: (a) add a private `bool m_from_mem_addr = false` flag set only by the `(string, size_t, size_t)` constructor; (b) in `load_external_mem_data()` throw `invalid_external_data` if `!m_from_mem_addr`; (c) alternatively, reject any `TensorProto` whose parsed location equals `ORT_MEM_ADDR` with a hard error inside the `TensorProto` constructor path: `if (m_data_location == ORT_MEM_ADDR) throw error::invalid_external_data{...};` at line ~22. This prevents any file-sourced model from ever reaching the pointer-reinterpret path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #419.
