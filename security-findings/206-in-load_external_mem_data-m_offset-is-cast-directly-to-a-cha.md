# Security finding #206: In `load_external_mem_data()`, `m_offset` is cast directly to a cha…

**Summary:** In `load_external_mem_data()`, `m_offset` is cast directly to a cha…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker who can supply or modify an ONNX file parsed by the OV ONNX frontend can achieve arbitrary memory read (information leak or process crash / DoS). By setting `offset` to an arbitrary non-null value and `data_length` to a desired read size, the memcpy will read from an attacker-chosen virtual address, either leaking memory contents or triggering a SIGSEGV. On platforms where the attacker can control what resides at the crafted address (e.g. via heap spray or a large model buffer), this could be escalated toward RCE.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX file on disk parsed by the OV ONNX frontend (e.g. via read_model() or the EP passing a model path to OV); the ORT_MEM_ADDR sentinel is meant to carry only in-process ORT pre-placed pointers, but there is no check that prevents a file-sourced TensorProto from asserting this location.

## Description / Root cause
In `load_external_mem_data()`, `m_offset` is cast directly to a char pointer at line 126 (`reinterpret_cast<char*>(m_offset)`) and then used as the source in `std::memcpy` at line 129. `m_offset` was populated at `tensor_external_data.cpp:24` by `std::stoull(entry.value())` from the ONNX proto's external-data 'offset' field — i.e., straight from a parsed file. The only guard (lines 121-124) rejects the case where `m_offset == 0 && m_data_length > 0` but allows any non-zero offset. There is zero check that the `*/_ORT_MEM_ADDR_/*` sentinel was placed by a legitimate in-process ORT pre-processing step rather than written into a file by an attacker.

**Validator analysis:** The flaw is real and the CWE-822 classification is accurate. The ORT_MEM_ADDR sentinel ('*/_ORT_MEM_ADDR_/*') is just a plain string in the ONNX TensorProto.external_data 'location' field, and 'offset' is an arbitrary uint64 parsed by std::stoull (line 24). For a file-loaded model m_tensor_place is nullptr, so Tensor::get_ov_constant() builds TensorExternalData from the untrusted proto (tensor.cpp:453) and, because the location matches ORT_MEM_ADDR (tensor.cpp:455), calls load_external_mem_data(). The only guard (lines 121-124) merely rejects offset==0 with length>0; any non-zero offset is accepted and reinterpret_cast to a pointer that memcpy reads from — yielding attacker-controlled arbitrary read / SIGSEGV (info leak / DoS), as stated. Impact assessment is sound; RCE is speculative but the read/crash primitive is solid. The proposed fix's second (more robust) variant is correct and sufficient: gate the ORT_MEM_ADDR branch in get_ov_constant() on m_tensor_place != nullptr and otherwise throw invalid_external_data, so a file-parsed proto can never reach the raw-pointer path. The session-flag variant works too but is more invasive; the m_tensor_place!=nullptr check is the minimal, targeted guard. openvinoEp is rejected because its genuine in-process tensor placement uses the TensorONNXPlace path (non-null m_tensor_place) and the EP does not contain or independently reach the nullptr-place file-sourced sentinel path.

## Exploit / Proof of Concept
Craft an ONNX file whose initializer tensor has `data_location='*/_ORT_MEM_ADDR_/*'`, `offset=<any non-zero value, e.g. 0x41414141>`, and `data_length=0x1000`. The file-parse constructor `TensorExternalData(const TensorProto&)` at tensor_external_data.cpp:19-30 stores these verbatim. In `Tensor::get_ov_constant()` (tensor.cpp:453), because `m_tensor_place == nullptr` (the regular file-load path sets it to nullptr at tensor.hpp:192-193), the file-sourced proto is used. At line 455 the location matches `ORT_MEM_ADDR`, so `load_external_mem_data()` is called. At line 126 `reinterpret_cast<char*>(0x41414141)` is formed; line 129 executes `memcpy` from that address → SIGSEGV / arbitrary read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
// (load_external_mem_data raw reinterpret_cast<char*>(m_offset) + memcpy).
//
// Pre-fix: convert_model() of a model whose initializer has
//   external_data location = "*/_ORT_MEM_ADDR_/*", offset = <bogus addr>, length > 0
// reaches load_external_mem_data() (tensor.cpp:455-456 with m_tensor_place==nullptr)
// and memcpy's from an attacker-chosen address -> ASan SEGV / arbitrary read.
// Post-fix: the ORT_MEM_ADDR branch is only reachable when m_tensor_place!=nullptr,
// so a file-sourced proto must throw ov::Exception (invalid_external_data) instead.
//
// NOTE: this requires a crafted .onnx fixture; emitted as a SKELETON.
#include "onnx_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: add binary fixture models/ort_mem_addr_file_sourced.onnx whose single
//       initializer carries external_data entries:
//         location = "*/_ORT_MEM_ADDR_/*"
//         offset   = "1094795585"   // 0x41414141, an unmapped address
//         length   = "4096"
//       (a file-sourced TensorProto, NOT placed by an in-process ORT session).
TEST(onnx_importer, DISABLED_ort_mem_addr_must_not_dereference_file_sourced_offset) {
    // Must be rejected at parse/convert time, never dereferenced.
    EXPECT_THROW(convert_model("ort_mem_addr_file_sourced.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_importer.*ort_mem_addr_must_not_dereference_file_sourced_offset' (remove DISABLED_ once the crafted .onnx fixture is added). Pre-fix expectation: AddressSanitizer SEGV on unknown address 0x000041414141 inside std::memcpy from TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129). Post-fix expectation: test passes because convert_model throws ov::Exception (invalid_external_data) for an ORT_MEM_ADDR location on a file-sourced proto.

## Suggested fix
Introduce a session-level or model-load-level flag that is set only when ORT is placing genuine in-process tensors via the TensorONNXPlace code path (tensor.cpp:449-452). In `load_external_mem_data()`, add an explicit guard that this flag is set before accepting the sentinel, e.g.: `OPENVINO_ASSERT(m_from_ort_session, "ORT_MEM_ADDR tensors are only valid for in-process ORT sessions, not file-sourced models");`. Alternatively — and more robustly — restrict the `ORT_MEM_ADDR` branch in `Tensor::get_ov_constant()` (tensor.cpp:455-456) so it is only reachable when `m_tensor_place != nullptr` (the in-process ORT path), and throw `invalid_external_data` if `m_tensor_place == nullptr` and the location is `ORT_MEM_ADDR`. This ensures a file-parsed proto can never trigger the raw-pointer dereference path.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #206.
