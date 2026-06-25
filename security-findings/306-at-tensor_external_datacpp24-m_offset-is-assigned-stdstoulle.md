# Security finding #306: At tensor_external_data.cpp:24, `m_offset` is assigned `std::stoull…

**Summary:** At tensor_external_data.cpp:24, `m_offset` is assigned `std::stoull…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker who supplies a crafted ONNX model file can read an arbitrary range of the loading process's virtual address space. When OpenVINO (or the OV ONNX frontend) is used in a server-side model-loading service, this is a critical information-disclosure primitive: the attacker can exfiltrate heap contents, stack data, cryptographic keys, session tokens, or other secrets resident in the process. The read result is returned as model weight data to the caller.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file (TensorProto.external_data["offset"]) deserialized into TensorExternalData::m_offset via std::stoull, then unconditionally cast to a pointer

## Description / Root cause
At tensor_external_data.cpp:24, `m_offset` is assigned `std::stoull(entry.value())` with no range or sanity validation. At line 126, `char* addr_ptr = reinterpret_cast<char*>(m_offset)` converts this fully attacker-controlled integer directly to a pointer. At line 129, `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` reads `m_data_length` bytes (also attacker-controlled, line 26) from the derived address. The only guard (lines 121-124) merely checks `m_offset != 0 && m_data_length != 0`, which does not restrict the address to any legitimate allocation.

**Validator analysis:** Confirmed for openvino: TensorExternalData(const TensorProto&) (cpp:19-36) populates m_data_location/m_offset/m_data_length purely from untrusted external_data entries with no validation; std::stoull can also throw on malformed input (separate DoS, not the cited issue). The dispatch in tensor.hpp get_external_data() picks load_external_mem_data() whenever data_location()==ORT_MEM_ADDR — and that string is itself attacker-supplied via the proto, so a file-loaded model (has_external_data() true => data_location==EXTERNAL, location entry == '*/_ORT_MEM_ADDR_/*') reaches the memcpy from a fully attacker-chosen address. No bounds, no mapping, no caller whitelist. CWE-822 untrusted pointer dereference / CWE-125 OOB read is accurate; impact (arbitrary in-process memory disclosure returned as tensor weights, and likely crash on unmapped pages) is accurate. The proposed fix is correct in spirit: part (2) — reject ORT_MEM_ADDR on the file/proto path — is the necessary and sufficient gate. The cleaner, simpler implementation is to dispatch to load_external_mem_data() ONLY when m_tensor_place != nullptr (the trusted ORT in-process handoff via the 3-arg ctor) in tensor.hpp:323-325, and otherwise throw invalid_external_data when a proto-derived TensorExternalData carries the ORT_MEM_ADDR marker; the part (1) per-call address whitelist is over-engineered and unnecessary if the proto path can never select this branch. For openvinoEp the code does not exist and the EP only supplies validated in-process pointers, so it is na.

## Exploit / Proof of Concept
Craft an ONNX model with a TensorProto whose `data_location` field (external_data key "location") is set to the string "*/_ORT_MEM_ADDR_/*" and whose "offset" field is set to a numeric string encoding a known/guessable process address (e.g., the base address of a shared library, whose ASLR slide can often be obtained by a prior information leak or brute-forced). Set "length" to any positive value (e.g., 4096). When this model is loaded via the ONNX frontend: (1) `TensorExternalData(*m_tensor_proto)` (tensor_external_data.cpp:19-36) parses `m_offset` and `m_data_length` from the proto strings; (2) the dispatch in tensor.hpp:324 fires because `ext_data.data_location() == detail::ORT_MEM_ADDR`; (3) `load_external_mem_data()` passes the two `!= 0` guards and executes `memcpy` from the attacker-specified address into an AlignedBuffer that is then returned as the tensor's weight data. No out-of-band authentication of the model or the address is performed at any point on this path.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822/CWE-125 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:24,126,129
// A file-loaded ONNX model must NOT be able to set external_data location to the
// ORT shared-memory marker "*/_ORT_MEM_ADDR_/*" and have offset reinterpret_cast to a
// raw pointer that is memcpy'd from (tensor_external_data.cpp:126,129).
//
// Pre-fix: convert_model() reaches load_external_mem_data() and performs memcpy from an
//          attacker-chosen address -> ASan SEGV / heap-buffer-overflow read (or arbitrary read).
// Post-fix: the proto path rejects ORT_MEM_ADDR and throws ov::Exception.
//
// This needs a crafted .onnx fixture, so it is a SKELETON: the binary model cannot be
// authored inline without protobuf tooling.

#include "onnx_import_test_helpers.hpp"   // TODO: confirm exact helper header in src/frontends/onnx/tests/
#include "gtest/gtest.h"
#include "common_test_utils/test_control.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: Create models/ort_mem_addr_arbitrary_read.onnx (or models/onnx/ ...) with a
//       TensorProto where:
//         data_location  = EXTERNAL
//         external_data: { key:"location", value:"*/_ORT_MEM_ADDR_/*" }
//                        { key:"offset",   value:"<some non-zero integer>" }
//                        { key:"length",   value:"4096" }
//       Place it where the test's model-path resolver (e.g. util::path_join /
//       TEST_ONNX_MODELS_DIRNAME) can find it.
TEST(onnx_external_data, reject_ort_mem_addr_from_file_loaded_model) {
    // TODO: replace with the project's model-loading helper used by onnx_import.in.cpp,
    //       e.g. convert_model("ort_mem_addr_arbitrary_read.onnx").
    EXPECT_THROW(convert_model("ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_external_data.reject_ort_mem_addr_from_file_loaded_model'. Pre-fix expected: ASan SEGV / 'heap-buffer-overflow READ' (arbitrary pointer read) inside TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129 memcpy), or process crash. Post-fix expected: test passes because the proto/file path rejects ORT_MEM_ADDR and throws ov::Exception. TODO: provide the crafted .onnx fixture described in the test comments.

## Suggested fix
The `ORT_MEM_ADDR` mode is a shared-memory hand-off protocol between ORT and its execution providers; it must never be reachable when loading an ONNX model from an untrusted file/stream. The fix has two parts: (1) In `load_external_mem_data()`, add a runtime guard that validates `m_offset` against a whitelist of addresses passed in by the trusted caller (e.g., a `std::set<std::pair<uintptr_t,size_t>> allowed_regions` parameter), and throw `invalid_external_data` if the range `[m_offset, m_offset+m_data_length)` falls outside every allowed region. (2) In the file-loading path (i.e., when `Tensor::get_external_data()` is reached because a TensorProto was parsed from disk or network), reject `ORT_MEM_ADDR` outright before dispatching: `if (ext_data.data_location() == detail::ORT_MEM_ADDR) throw error::invalid_external_data{"ORT_MEM_ADDR is not permitted for file-loaded models"};`. This ensures the code path is only reachable when ORT explicitly constructs a `TensorExternalData` with a validated in-process pointer via the `TensorExternalData(location, offset, size)` constructor (tensor_external_data.cpp:37-41).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #306.
