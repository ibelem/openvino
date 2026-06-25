# Security finding #443: `m_data_length = std::stoull(entry.value())` applies no upper-bound…

**Summary:** `m_data_length = std::stoull(entry.value())` applies no upper-bound…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Root cause of both the OOM/allocation-overflow (CWE-789) and the untrusted-pointer-dereference (CWE-822) findings above. Fixing it at the parse site is the most reliable mitigation since it prevents the value from propagating into any consumer.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:26` — `TensorExternalData::TensorExternalData(const TensorProto&)()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file (attacker-supplied protobuf) → m_data_length integer field

## Description / Root cause
`m_data_length = std::stoull(entry.value())` applies no upper-bound validation. `uint64_t` can hold values up to 18446744073709551615. Both subsequent allocation paths that use this field without a file-size guard (`load_external_mem_data`, and indirectly any caller that uses the size for arithmetic) inherit this unbounded value. The file-based paths protect themselves locally (lines 83–85, 53–56) but `load_external_mem_data` does not, and the root cause is the absence of validation at parse time.

**Validator analysis:** Confirmed real and reachable from openvino's ONNX-model trust boundary. Tensor::get_external_data (tensor.hpp:316-332) constructs TensorExternalData(*m_tensor_proto) (the line-19/26 ctor) for any initializer whose data_location==EXTERNAL (tensor.hpp:312-313). The location string is read verbatim from the attacker's protobuf (cpp:21-22), so a malicious model can set location='*/_ORT_MEM_ADDR_/*'. That makes ext_data.data_location()==ORT_MEM_ADDR (tensor.hpp:324) and dispatches to load_external_mem_data, which — unlike the file paths (cpp:53-56, 83-85) — has NO file-size/bounds guard. There m_offset is reinterpret_cast to a raw pointer (cpp:126) and m_data_length bytes are memcpy'd from it (cpp:129) after an attacker-sized AlignedBuffer allocation (cpp:127). So line 26's missing upper bound is a genuine root cause: a huge length yields OOM/std::bad_alloc (CWE-789), and an arbitrary m_offset+length yields an arbitrary-address read (CWE-822). The vulnType (CWE-20) and impact (root cause of both downstream issues) are accurate. The proposedFix is only PARTIALLY sufficient: a MAX_TENSOR_SIZE_BYTES bound on m_data_length blocks the OOM allocation but a 16GB memcpy from a forged pointer (cpp:129) is still an arbitrary read — the decisive fix is the second clause the author already notes: reject location==ORT_MEM_ADDR (and reinterpret_cast<char*>(m_offset)) for any TensorProto that originates from a parsed file (m_tensor_place==nullptr), since ORT_MEM_ADDR is only legitimate for in-memory weights handed over by the ORT EP, never from a serialized model file. Recommend gating load_external_mem_data behind an explicit 'data came from trusted in-memory source' flag rather than trusting the on-disk location string, plus the length bound.

## Exploit / Proof of Concept
Supply an ONNX model whose `external_data` `length` field is `"18446744073709486080"`. `std::stoull` parses it to `0xFFFFFFFFFFFF0000` without error; the resulting `m_data_length` is stored and later used unchecked in `load_external_mem_data` at line 127.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for tensor_external_data.cpp:26 (unbounded std::stoull length)
// and the unchecked sink load_external_mem_data() at cpp:126-129.
// Pre-fix: a parsed ONNX initializer with external_data location='*/_ORT_MEM_ADDR_/*',
//   a forged offset and a huge length reaches reinterpret_cast<char*>(m_offset) + memcpy,
//   producing an arbitrary-address read / std::bad_alloc (ASan: SEGV or allocation-size-too-big).
// Post-fix: the frontend must reject ORT_MEM_ADDR (and/or oversized length) for a
//   file-originated TensorProto, so convert_model throws ov::Exception cleanly.
//
// Style follows onnx_import.in.cpp; lives in the ov_onnx_frontend_tests target.
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

// TODO: add a crafted fixture models/ort_mem_addr_external_length.onnx whose single
//   initializer has data_location=EXTERNAL and external_data entries:
//     location = "*/_ORT_MEM_ADDR_/*"
//     offset   = "4096"                       // forged raw address
//     length   = "18446744073709486080"       // 0xFFFFFFFFFFFF0000, parsed by std::stoull
//   (No raw_data; data_type FLOAT, dims small.) A pure .onnx fixture is required because
//   the trigger is the serialized external_data table, not a runtime API arg.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_length_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_external_length.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_length_rejected*. Pre-fix expectation: ASan reports 'SEGV on unknown address' (memcpy from reinterpret_cast<char*>(m_offset) at tensor_external_data.cpp:129) or 'requested allocation size exceeds maximum supported size' (AlignedBuffer at cpp:127); test fails because no ov::Exception is thrown. Post-fix: convert_model throws ov::Exception (invalid_external_data) and the test passes. NOTE: requires adding the crafted models/ort_mem_addr_external_length.onnx fixture described in the TODO.

## Suggested fix
After `std::stoull`, add a sanity bound: `if (m_data_length > MAX_TENSOR_SIZE_BYTES) { OPENVINO_THROW("external_data length exceeds maximum"); }` where `MAX_TENSOR_SIZE_BYTES` is a compile-time constant (e.g. `1ULL << 34` for 16 GB). Separately, validate that `m_data_location` is never set to `ORT_MEM_ADDR` from a file-originated `TensorProto` (see CWE-822 fix above).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #443.
