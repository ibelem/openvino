# Security finding #126: ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") is declared as a public const s…

**Summary:** ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") is declared as a public const s…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** The ORT_MEM_ADDR check provides a false sense of security. Developers may believe that path is unreachable from untrusted input, but it is trivially reachable. This amplifies the severity of CWE-822 above to a remotely-exploitable ONNX-parse attack requiring no special knowledge beyond reading the public OpenVINO header.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.hpp:91`
**Validated for repos:** openvino
**Trust boundary:** ONNX model 'location' string → TensorExternalData constructor → load_external_mem_data() guard

## Description / Root cause
ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") is declared as a public const std::string in the header, visible to any consumer of the API and in all compiled ONNX model handling code. The check at load_external_mem_data():117 (`if (m_data_location != ORT_MEM_ADDR) throw`) was designed as a sentinel meaning 'this tensor was placed by ORT in shared memory', but because the value is a non-secret, publicly-specified string, any ONNX model file can reproduce it to unlock the arbitrary-pointer-dereference code path. There is no runtime attestation that the tensor actually originates from an ORT in-process hand-off.

**Validator analysis:** Confirmed real and reachable for openvino. The proto-path TensorExternalData ctor (tensor_external_data.cpp:21-22,24) copies the attacker-controlled 'location' and 'offset' verbatim; has_external_data() (tensor.hpp:312-313) is true for any EXTERNAL tensor, and get_external_data() dispatches on the public sentinel string (tensor.hpp:324) with no provenance check, landing in load_external_mem_data() which casts m_offset to a pointer and memcpy's m_data_length bytes (tensor_external_data.cpp:126-129). The true underlying defect is CWE-822 (untrusted-pointer dereference / arbitrary read, potential info-leak or crash); the CWE-20 framing here is accurate as the root cause: the only guard (m_data_location==ORT_MEM_ADDR) discriminates on a non-secret, public-header constant, so it provides no trust attestation. Impact is correctly characterized as a remotely-triggerable ONNX-parse arbitrary read. The proposed fix is correct and sufficient in direction: the sentinel/in-memory-address path must be gated on a provenance flag set ONLY by the internal ORT TensorONNXPlace/tensor_place constructor, and the TensorProto (file) constructor must reject ORT_MEM_ADDR. A clean equivalent: in get_external_data()/has_external_data() only allow the ORT_MEM_ADDR branch when m_tensor_place != nullptr (the in-process hand-off path), and have the TensorProto ctor throw on location==ORT_MEM_ADDR. Either form closes the file-based reach while preserving the legitimate ORT shared-memory use.

## Exploit / Proof of Concept
Read tensor_external_data.hpp to learn the sentinel string, then set external_data[location]="*/_ORT_MEM_ADDR_/*" in any ONNX model. Because ORT_MEM_ADDR is a compile-time constant in a public API header, no reverse engineering is needed.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-20/CWE-822 at:
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:116-129
//   openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:324
// A file-based ONNX model whose external_data 'location' == "*/_ORT_MEM_ADDR_/*"
// and whose 'offset' is an attacker-chosen integer currently flows into
// reinterpret_cast<char*>(m_offset) + memcpy => arbitrary-pointer read.
// Pre-fix: ASan SEGV / heap-buffer-overflow (or silent OOB read) during import.
// Post-fix: import must reject the model with ov::Exception because the
// ORT_MEM_ADDR sentinel is not permitted in a file-based TensorProto.
//
// Harness: ov_onnx_frontend_tests (style of onnx_import.in.cpp), gtest + ASan.
//
// SKELETON: triggering requires a crafted .onnx binary fixture that cannot be
// authored as plain source here.  See TODOs.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: place a crafted model at
//   onnx/frontend/tests/models/external_data/ort_mem_addr_sentinel_rejected.onnx
// containing one initializer with:
//   data_location = EXTERNAL
//   external_data: { key:"location", value:"*/_ORT_MEM_ADDR_/*" },
//                  { key:"offset",   value:"4096" },   // arbitrary address
//                  { key:"length",   value:"64" }
// TODO: confirm the exact OPENVINO_TEST namespace/macro and model-loader helper
//       by reading onnx_import.in.cpp in this test tree before use.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_sentinel_rejected) {
    // Pre-fix this convert_model dereferences reinterpret_cast<char*>(4096).
    // Post-fix it must throw because the file-based path may not use ORT_MEM_ADDR.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_sentinel_rejected.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_sentinel_rejected*. Expected pre-fix: AddressSanitizer SEGV / unknown-address read inside detail::TensorExternalData::load_external_mem_data (memcpy from reinterpret_cast<char*>(m_offset)). Expected post-fix: test passes (convert_model throws ov::Exception rejecting the ORT_MEM_ADDR sentinel in a file-based model).

## Suggested fix
Remove the reliance on a publicly-known string as the only discriminator. Instead, gate the load_external_mem_data() call at get_ov_constant():455 on a runtime flag (e.g., a bool m_is_ort_shared_memory set only by the internal ORT tensor-place constructor path, never by the TensorProto constructor path). The TensorProto constructor (lines 19-36) should unconditionally reject ORT_MEM_ADDR in the 'location' key: 'if (m_data_location == ORT_MEM_ADDR) throw error::invalid_external_data{"ORT_MEM_ADDR sentinel not allowed in file-based models"};'.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #126.
