# Security finding #385: At line 287, `ext_data_offset = std::stoull(entry.value())` convert…

**Summary:** At line 287, `ext_data_offset = std::stoull(entry.value())` convert…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Any downstream code that reads `tensor_meta_info.m_tensor_data` (and its paired `m_tensor_data_size`) treats the stored value as a valid buffer pointer. This enables: (1) arbitrary read of any mapped memory page at attacker-chosen address and length — information disclosure of secrets in the process (model weights, tokens, private keys); (2) crash/DoS when the address is unmapped; (3) on a 32-bit platform or with a value aligned to a valid page, potential exploitation. Affects any user of the OpenVINO ONNX frontend that loads models from untrusted sources.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:287` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX `external_data.offset` and `external_data.location` fields from an untrusted model file, parsed before the ORT_MEM_ADDR string check

## Description / Root cause
At line 287, `ext_data_offset = std::stoull(entry.value())` converts the attacker-controlled ONNX `external_data.offset` string into a raw `uint64_t` with no range or validity checks. At line 297, inside the `ORT_MEM_ADDR` branch (line 294), it is unconditionally cast to a pointer: `tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(ext_data_offset)`. The string `ORT_MEM_ADDR = "*/_ORT_MEM_ADDR_/*"` (tensor_external_data.hpp:91) is just a plain literal that any ONNX file can supply. There is no trusted-caller gate, no address-range check, no alignment check, and no null check before storing the result. The function returns `true` at line 300, causing `extract_tensor_meta_info` (line 396) to return the struct with the unchecked pointer to the caller.

**Validator analysis:** Confirmed in openvino. ORT_MEM_ADDR ('*/_ORT_MEM_ADDR_/*') is a constant string literal (tensor_external_data.hpp:91); any ONNX file can set external_data location to it and supply an arbitrary decimal offset. At graph_iterator_proto.cpp:287 std::stoull converts the offset to uint64_t, and at :297 it is unconditionally reinterpret_cast to uint8_t* and stored as m_tensor_data with attacker-set m_tensor_data_size (:298). The branch returns true at :300 and short-circuits all file/path/size validation at :302-311. Downstream constant materialization then reads m_tensor_data_size bytes from the chosen address -> arbitrary read / DoS. CWE-822 Untrusted Pointer Dereference and the info-disclosure/DoS impact are accurate. There is NO surrounding mitigation: extract_tensor_meta_info:394-396 unconditionally enters this path for data_location==EXTERNAL and the frontend cannot distinguish an ORT-managed model from a file-loaded one. Proposed fix option (1) — a trusted-source flag set only by the ORT interop construction path, guarding the ORT_MEM_ADDR branch before the cast — is the correct and sufficient remedy. Option (2) (range/non-zero checks) is insufficient because any plausibly-mapped address still yields an arbitrary read; it should not be relied upon. openvinoEp is na: the vulnerable code and the trust decision both reside in the openvino frontend, and the EP is the legitimate trusted user of the marker.

## Exploit / Proof of Concept
Craft an ONNX protobuf (`TensorProto`) with `data_location = EXTERNAL` and two `external_data` entries: `key=location, value=*/_ORT_MEM_ADDR_/*` and `key=offset, value=<target address as decimal, e.g. 4096>` plus `key=length, value=65536`. Load this via `GraphIteratorProto::initialize()`. Because `ext_location` matches `detail::ORT_MEM_ADDR` at line 294, the function bypasses all file-based validation (lines 303–311 are skipped) and stores `(uint8_t*)4096` with size 65536 in `m_tensor_data`. When the inference engine subsequently copies or iterates over the tensor's raw data, it reads 65536 bytes from address 4096 — a fully attacker-directed arbitrary memory read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:287-300
// (extract_tensor_external_data ORT_MEM_ADDR branch).
//
// Pre-fix: a model whose TensorProto has data_location=EXTERNAL with
//   external_data: location="*/_ORT_MEM_ADDR_/*", offset="4096", length="65536"
// causes the frontend to reinterpret_cast(4096) into m_tensor_data and accept it
// (returns true at :300), so convert_model() succeeds and later constant folding
// dereferences the bogus pointer (ASan: SEGV / heap-buffer / unknown-address READ).
// Post-fix (trusted-source gate added before :297): convert_model() must REJECT the
// untrusted ORT_MEM_ADDR model with ov::Exception.
//
// TODO(fixture): add a crafted protobuf model file
//   onnx/models/ort_mem_addr_untrusted.onnx (or .prototxt) containing a single
//   initializer TensorProto with:
//     data_location: EXTERNAL
//     external_data { key: "location" value: "*/_ORT_MEM_ADDR_/*" }
//     external_data { key: "offset"   value: "4096" }
//     external_data { key: "length"   value: "65536" }
//   Place it under the frontend test models dir resolved by
//   util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, ...}).
// TODO(symbols): confirm the exact convert_model/exception helpers from
//   src/frontends/onnx/tests/onnx_import.in.cpp (FrontEndManager / onnx fixture).

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // ov::frontend::onnx::tests::convert_model

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, ort_mem_addr_from_untrusted_model_is_rejected) {
    // Must throw once the ORT_MEM_ADDR branch is gated to trusted callers only.
    EXPECT_THROW(convert_model("ort_mem_addr_untrusted.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='onnx_external_data.ort_mem_addr_from_untrusted_model_is_rejected'. Pre-fix expectation: no exception thrown (test FAILS) and, if the bogus pointer is dereferenced during constant materialization, ASan reports 'SEGV on unknown address 0x000000001000' / heap-buffer-overflow READ. Post-fix expectation: convert_model throws ov::Exception and the test PASSES. Requires adding the crafted ort_mem_addr_untrusted.onnx fixture noted in the test's TODO.

## Suggested fix
The ORT shared-memory path must be restricted to a trusted API entry point, not inferred from a string in an untrusted file. Options: (1) Add a `bool m_ort_trusted_source` flag to `GraphIteratorProto` that is set only when the model is constructed by the ORT interop layer (not from a file), and guard the `ORT_MEM_ADDR` branch with `if (!graph_iterator->is_ort_trusted_source()) throw std::runtime_error("ORT_MEM_ADDR not allowed for untrusted models");`. (2) At minimum, validate that `ext_data_offset` is a plausible user-space address (non-zero, within a configured range) and that `ext_data_length > 0`. The fix must be at the entry of the `ORT_MEM_ADDR` branch (before line 297), not after the cast.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #385.
