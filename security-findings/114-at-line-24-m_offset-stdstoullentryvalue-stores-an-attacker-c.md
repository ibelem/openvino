# Security finding #114: At line 24, `m_offset = std::stoull(entry.value())` stores an attac…

**Summary:** At line 24, `m_offset = std::stoull(entry.value())` stores an attac…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read of process memory up to `m_data_length` bytes starting at any attacker-specified virtual address. Can leak secrets (cryptographic keys, model weights, heap/stack contents) or crash the process with an access violation if the address is unmapped. Triggered by any user loading a crafted ONNX model through the OpenVINO ONNX frontend.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied ONNX model file → TensorProto parser → TensorExternalData constructor → load_external_mem_data()

## Description / Root cause
At line 24, `m_offset = std::stoull(entry.value())` stores an attacker-controlled integer from the ONNX model's `external_data` key-value pairs without any range or address validation. When `m_data_location == ORT_MEM_ADDR` (line 117 passes), the only guard (lines 121–125) is `m_offset != 0 && m_data_length != 0`. Line 126 then does `char* addr_ptr = reinterpret_cast<char*>(m_offset)`, treating the attacker-supplied integer as a raw process address, and line 129 executes `std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length)` — reading from that arbitrary address.

**Validator analysis:** vulnType CWE-822 (Untrusted Pointer Dereference) is accurate: m_offset, an attacker-controlled uint64 from the model's external_data 'offset' entry, is reinterpret_cast to a raw process pointer (line 126) and memcpy'd from (line 129). The impact (arbitrary read up to m_data_length bytes / crash on unmapped address) is correct; the guard at lines 121-125 only rejects offset==0/length==0, never validating the address. Reachability is genuine because has_external_data() (tensor.hpp:312-313) returns true for any DataLocation_EXTERNAL tensor, and both dispatch sites (tensor.cpp:455 in get_ov_constant and tensor.hpp:324 in get_external_data) select the ORT_MEM_ADDR branch purely by string equality, independent of whether data came from a trusted in-process place or an untrusted file. The proposed fix (guard the ORT_MEM_ADDR branch with m_tensor_place != nullptr) is correct and is the right minimal mitigation, BUT it is INCOMPLETE as written: it must be applied to BOTH dispatch sites — tensor.cpp:455 AND the identical unguarded dispatch in tensor.hpp:324 (get_external_data) — otherwise the arbitrary read is still reachable for integer/string typed initializers. The constructor-side alternative ('throw in TensorExternalData(const TensorProto&) when location==ORT_MEM_ADDR') is actually the cleaner single-point fix since that constructor is ONLY used on the file path, and it covers both call sites at once; I recommend that variant. A file-size bound on offset alone does not fix this because the offset is interpreted as a pointer, not a file offset, on this branch.

## Exploit / Proof of Concept
Craft an ONNX model with a tensor whose `external_data` field contains: `key="location", value="*/_ORT_MEM_ADDR_/*"` and `key="offset", value="<target_address_as_decimal>"` and `key="length", value="4096"`. When `Tensor::get_ov_constant()` (tensor.cpp:453) constructs `TensorExternalData(*m_tensor_proto)`, `m_offset` gets the attacker integer. tensor.cpp:455 evaluates `ext_data.data_location() == detail::ORT_MEM_ADDR` as true and calls `load_external_mem_data()`. The non-zero check at line 123 passes trivially; line 126 casts the integer to `char*`, and line 129 reads up to 4096 bytes from that process address into `aligned_memory`, which is then returned as the tensor constant and potentially serialized or logged.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 untrusted pointer dereference in the ONNX frontend.
// Unchecked code: targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126,129
//   char* addr_ptr = reinterpret_cast<char*>(m_offset); std::memcpy(dst, addr_ptr, m_data_length);
// Reached via targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:455 (and tensor.hpp:324),
// which select load_external_mem_data() by a bare string compare against detail::ORT_MEM_ADDR
// (= "*/_ORT_MEM_ADDR_/*") with no m_tensor_place != nullptr guard.
//
// This test loads a crafted model whose initializer uses data_location=EXTERNAL with
// external_data entries location="*/_ORT_MEM_ADDR_/*", offset=<bogus address>, length=4096.
// PRE-FIX: dispatch enters load_external_mem_data(), reinterpret_casts the attacker offset to a
//          pointer and memcpy-reads -> ASan SEGV / arbitrary read (test crashes, no throw).
// POST-FIX: the file path must NOT reach the ORT_MEM_ADDR branch; convert_model must reject the
//           model with ov::Exception (error::invalid_external_data).
//
// NOTE: this needs a crafted binary .onnx fixture, so it is emitted as a SKELETON.
// Place in the style of openvino/src/frontends/onnx/tests/onnx_import.in.cpp.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // get_models_dir(), convert_model() helpers used across onnx tests
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, reject_ort_mem_addr_in_file_based_model) {
    // TODO: add a crafted fixture model file at
    //   openvino/src/frontends/onnx/tests/models/external_data/ort_mem_addr_arbitrary_read.onnx
    // containing one float initializer with:
    //   tensor.data_location = EXTERNAL
    //   external_data: {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
    //                  {key:"offset",   value:"<decimal bogus virtual address, e.g. 0x4141414141414141>"}
    //                  {key:"length",   value:"4096"}
    // TODO: confirm the exact convert_model() helper name/signature from onnx_utils.hpp in the test tree.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_arbitrary_read.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.reject_ort_mem_addr_in_file_based_model . Pre-fix expected: AddressSanitizer SEGV / 'unknown-crash' on a wild read inside std::memcpy in TensorExternalData::load_external_mem_data (tensor_external_data.cpp:129) — the EXPECT_THROW never gets a clean throw. Post-fix expected: convert_model throws ov::Exception (error::invalid_external_data, 'ORT_MEM_ADDR not permitted in file-based models') and the test passes. Requires authoring the crafted .onnx fixture noted in the TODO.

## Suggested fix
The `ORT_MEM_ADDR` path is an in-process inter-framework memory-sharing facility that was never meant to be reachable from a file-based ONNX model. Fix in `tensor.cpp`: only allow the `ORT_MEM_ADDR` branch when `m_tensor_place != nullptr` (the trusted programmatic path), i.e. change the dispatch at lines 449–461 so that when `m_tensor_place == nullptr` (data came from an ONNX file), a `data_location() == ORT_MEM_ADDR` result is treated as `error::invalid_external_data` rather than dispatching to `load_external_mem_data()`. Additionally, in the constructor `TensorExternalData(const TensorProto&)` at line 24, reject any offset whose numeric value cannot represent a valid file offset (e.g. exceeds a reasonable file-size bound) to prevent the integer from silently becoming a pointer. Concretely: add `if (m_data_location == ORT_MEM_ADDR) { throw error::invalid_external_data{"ORT_MEM_ADDR not permitted in file-based models"}; }` at the end of the protobuf constructor, or guard the `ext_data.data_location() == detail::ORT_MEM_ADDR` branch at tensor.cpp:455 with `m_tensor_place != nullptr`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #114.
