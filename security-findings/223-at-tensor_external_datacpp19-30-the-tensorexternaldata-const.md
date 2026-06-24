# Security finding #223: At tensor_external_data.cpp:19-30, the TensorExternalData construct…

**Summary:** At tensor_external_data.cpp:19-30, the TensorExternalData construct…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference / CWE-125: Out-of-bounds Read
**Severity / Impact:** Arbitrary process memory read. An attacker supplying a crafted ONNX model can cause the OpenVINO ONNX frontend to read an arbitrary number of bytes from any attacker-specified virtual address in the process. The read data is returned as tensor initializer content (via get_external_data()), making this an information-disclosure primitive that can leak secrets (keys, heap pointers for ASLR bypass, etc.) depending on the host application. May also crash the process if the target address is unmapped (DoS).
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file → TensorExternalData(const TensorProto&) constructor → load_external_mem_data() memcpy

## Description / Root cause
At tensor_external_data.cpp:19-30, the TensorExternalData constructor parsing a model-file-sourced TensorProto unconditionally accepts any `location` string, including the ORT sentinel `"*/_ORT_MEM_ADDR_/*"` (ORT_MEM_ADDR, defined in tensor_external_data.hpp:91), and parses the `offset` field via `std::stoull()` into `m_offset` (a uint64_t) with no validation. In load_external_mem_data() (line 121), the sole guard is `bool is_valid_buffer = m_offset && m_data_length`, which is true for any non-zero attacker-supplied pair. There is no pointer-range or address-space validation. Line 126 casts `m_offset` to `char*` (`reinterpret_cast<char*>(m_offset)`), and line 129 executes `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)`, reading `m_data_length` bytes from the attacker-controlled address.

**Validator analysis:** The flaw is real and accurately categorized as CWE-822/CWE-125 arbitrary-address read. Data flow confirmed: an untrusted ONNX model with a tensor whose data_location==EXTERNAL and external_data {location="*/_ORT_MEM_ADDR_/*", offset=<addr>, length=N} causes get_external_data() (tensor.hpp:317-331) to take the m_tensor_proto branch (m_tensor_place==nullptr), build a TensorExternalData from the proto (cpp:19-36, no validation of location/offset), and dispatch to load_external_mem_data() (cpp:116-134). The only guard, `m_offset && m_data_length` (cpp:121), is satisfied by any non-zero attacker pair, so cpp:126 reinterpret_cast<char*>(m_offset) and cpp:129 memcpy read N bytes from the attacker-chosen address into a buffer returned as initializer content — an info-leak (and DoS on unmapped pages). No upstream mitigation: load_external_mmap_data/load_external_data perform file_size bounds checks, but the ORT_MEM_ADDR path has none, and nothing rejects the sentinel when it comes from a parsed file. The proposed fix (an m_from_file flag set only in the TensorProto constructor, throwing invalid_external_data at the top of load_external_mem_data) is correct and sufficient; equivalently, only construct an ORT_MEM_ADDR TensorExternalData on the m_tensor_place!=nullptr branch. Impact statement is accurate. The defect is purely an openvino-frontend bug; the EP boundary does not turn file offsets into the dereferenced address, hence openvinoEp rejected. Reproduction requires a crafted binary .onnx fixture so a self-contained programmatic test is not fully achievable here — skeleton provided.

## Exploit / Proof of Concept
Craft a .onnx model with a tensor initializer whose `data_location` is set to EXTERNAL and whose `external_data` block contains: `key="location", value="*/_ORT_MEM_ADDR_/*"` (matching ORT_MEM_ADDR exactly), `key="offset", value="<target address as decimal integer>"`, `key="length", value="4096"`. When OpenVINO loads this model, Tensor::get_external_data() (tensor.hpp:322) constructs a TensorExternalData from the untrusted TensorProto (m_tensor_place==nullptr path), then at tensor.hpp:324 routes to load_external_mem_data() because data_location()==ORT_MEM_ADDR. The guard at line 123 does not fire (both offset and length are non-zero → is_valid_buffer=true), so execution reaches line 126 casting the offset integer to char*, and line 129 memcpy-reads 4096 bytes from the target address into the returned buffer, which the caller exposes as tensor data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in tensor_external_data.cpp:126-129
// (TensorExternalData::load_external_mem_data), reachable via core/tensor.hpp:322-325.
// Pre-fix: loading a model whose initializer has data_location=EXTERNAL and
//   external_data{location="*/_ORT_MEM_ADDR_/*", offset=<addr>, length=N} causes
//   reinterpret_cast<char*>(m_offset) + memcpy => arbitrary-address read (ASan: SEGV/heap-buffer-overflow).
// Post-fix: the file-sourced ORT_MEM_ADDR sentinel must be rejected with ov::Exception
//   (error::invalid_external_data), so convert_model() throws instead of dereferencing.
//
// Lives in: openvino/src/frontends/onnx/tests/onnx_import.in.cpp style file,
// built as target ov_onnx_frontend_tests.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // get_model_path / convert_model helpers

using namespace ov::frontend::onnx::tests;

// TODO: Provide a crafted fixture model 'ort_mem_addr_external_data.onnx' under
//       onnx/models/ whose single initializer has:
//         data_location = EXTERNAL
//         external_data: location="*/_ORT_MEM_ADDR_/*", offset="4096", length="4096"
//       (offset is a bogus low address; ANY non-zero offset triggers the cast+memcpy pre-fix).
//       A protobuf builder cannot be inlined here without the ONNX_NAMESPACE headers and
//       a serialize step, so the binary fixture is required.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_from_file_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_external_data.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_from_file_rejected*. Expected PRE-fix: ASan reports SEGV / out-of-bounds read inside TensorExternalData::load_external_mem_data (memcpy at tensor_external_data.cpp:129) when dereferencing reinterpret_cast<char*>(m_offset); test fails (no ov::Exception). Expected POST-fix: convert_model throws ov::Exception (error::invalid_external_data) for the file-sourced ORT_MEM_ADDR sentinel and the EXPECT_THROW passes. Requires the crafted ort_mem_addr_external_data.onnx fixture noted in the TODO.

## Suggested fix
The ORT_MEM_ADDR path must only be reachable when the TensorProto originates from in-process shared memory (not from a file). Add a boolean flag `m_from_file` to TensorExternalData and set it to true in the TensorProto-parsing constructor (lines 19-36). In load_external_mem_data(), add a guard at the top: `if (m_from_file) throw error::invalid_external_data{*this};` — this makes the sentinel completely inaccessible to model-file-sourced tensors. Alternatively, perform the ORT_MEM_ADDR check only in the `m_tensor_place != nullptr` branch of Tensor::get_external_data() (tensor.hpp:318-321) and never construct a TensorExternalData with ORT_MEM_ADDR from a TensorProto. Either way, never allow a file-parsed `offset` value to be dereferenced as a raw pointer.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #223.
