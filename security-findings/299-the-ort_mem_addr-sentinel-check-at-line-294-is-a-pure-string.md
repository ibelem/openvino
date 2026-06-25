# Security finding #299: The ORT_MEM_ADDR sentinel check at line 294 is a pure string equali…

**Summary:** The ORT_MEM_ADDR sentinel check at line 294 is a pure string equali…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** An attacker supplying a crafted ONNX file with `external_data[location] = "*/_ORT_MEM_ADDR_/*"` and `external_data[offset] = <any integer>` causes an attacker-controlled integer to be treated as a memory address and stored as the tensor data pointer. When this pointer is subsequently dereferenced (during model compilation/inference), it results in an arbitrary read from attacker-specified memory — information disclosure or crash (DoS). With careful alignment it could enable remote code execution via type confusion.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294` — `extract_tensor_external_data (anonymous namespace)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX file → GraphIteratorProto::initialize(const std::filesystem::path&) → protobuf-parsed TensorProto::external_data fields

## Description / Root cause
The ORT_MEM_ADDR sentinel check at line 294 is a pure string equality on `ext_location`, which is set from the protobuf field `entry.value()` (line 285) with no caller-context flag distinguishing file-load from legitimate in-process ORT injection. When taken, line 297 performs `reinterpret_cast<uint8_t*>(ext_data_offset)` where `ext_data_offset` was parsed from the untrusted protobuf `offset` field (line 287) and stores the resulting arbitrary pointer into `tensor_meta_info.m_tensor_data`. There is no flag on `GraphIteratorProto` that would block this branch when entered via `initialize(path)` (file-load path).

**Validator analysis:** Confirmed: GraphIteratorProto::initialize(const std::filesystem::path&) (line 504) parses an untrusted ONNX file (ParseFromIstream, line 511), and reset() (line 560) iterates initializers calling extract_tensor_meta_info->extract_tensor_external_data for any EXTERNAL tensor (line 394-396). At line 294 the ORT_MEM_ADDR sentinel (defined tensor_external_data.hpp:91) is matched by a plain string equality against the protobuf-supplied location field (line 285), with NO flag distinguishing the file-load path from legitimate ORT in-process injection. The header (graph_iterator_proto.hpp) confirms no such flag exists. When matched, line 297 reinterpret_casts the attacker-parsed offset (std::stoull, line 287) into a uint8_t* stored as m_tensor_data; this pointer is later dereferenced when the initializer's constant data is materialized during conversion -> arbitrary read / type confusion. CWE-822 Untrusted Pointer Dereference is accurate; impact (info disclosure/DoS, potential RCE via type confusion) is plausible and not overstated. The proposed fix (gate the ORT_MEM_ADDR branch on an m_is_ort_inprocess flag set only by initialize(shared_ptr<ModelProto>) and throw on the file-load path) is correct and sufficient; equivalently the file-load path should simply never honor the ORT_MEM_ADDR sentinel. Note std::stoull at lines 287/289 can also throw on non-numeric values, but that is caught by the surrounding try/catch in initialize(path) and is a separate, benign DoS, not the pointer issue. EP verdict is rejected because the defect is OpenVINO-internal and the attacker-controlled trigger is the core file-load path, not the EP wrapper.

## Exploit / Proof of Concept
Craft an ONNX model with one initializer tensor whose `data_location` is EXTERNAL, whose `external_data` repeated field contains `{key: "location", value: "*/_ORT_MEM_ADDR_/*"}` and `{key: "offset", value: "0xdeadbeef00"}`. Load the model via `ov::Core::read_model(path)` → `GraphIteratorProto::initialize(path)` → `reset()` → `extract_tensor_meta_info` → `extract_tensor_external_data`. The branch at line 294 is entered, `ext_data_offset = 0xdeadbeef00`, and `tensor_meta_info.m_tensor_data = reinterpret_cast<uint8_t*>(0xdeadbeef00)` is stored. When the framework later reads tensor data from that pointer, it dereferences the attacker-controlled address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in
// openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:294-297
// Pre-fix: a file-loaded ONNX model whose initializer has external_data
// location == "*/_ORT_MEM_ADDR_/*" and an arbitrary 'offset' causes that
// integer to be reinterpret_cast into m_tensor_data (an arbitrary pointer),
// which is dereferenced later during conversion (ASan: SEGV/arbitrary read).
// Post-fix: the file-load path rejects the ORT_MEM_ADDR sentinel and throws.
//
// This test is a SKELETON: it needs a crafted .onnx fixture because the
// trigger is a malformed binary TensorProto, which cannot be expressed in the
// gtest source alone.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: produce models/ort_mem_addr_sentinel.onnx with:
//   graph.initializer[0].data_location = EXTERNAL
//   external_data: {key:"location", value:"*/_ORT_MEM_ADDR_/*"},
//                  {key:"offset",   value:"0xdeadbeef00"},
//                  {key:"length",   value:"16"}
// Place under the onnx frontend test models dir (see onnx_utils.hpp helpers).
OPENVINO_TEST(${FRONTEND_NAME}_onnx, ort_mem_addr_sentinel_rejected_on_file_load) {
    // TODO: confirm the exact convert_model helper/signature used by this
    // test tree (e.g. convert_model("ort_mem_addr_sentinel.onnx")).
    EXPECT_THROW(convert_model("ort_mem_addr_sentinel.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (built with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*ort_mem_addr_sentinel_rejected_on_file_load*. Pre-fix expectation: ASan reports SEGV / heap-buffer-overflow (arbitrary read) when m_tensor_data == 0xdeadbeef00 is dereferenced during constant materialization, OR the EXPECT_THROW fails because no exception is raised. Post-fix expectation: convert_model throws ov::Exception (the file-load path rejects the ORT_MEM_ADDR sentinel) and the test passes. TODO: add the crafted models/ort_mem_addr_sentinel.onnx fixture before this compiles/runs.

## Suggested fix
Add a boolean flag `m_is_ort_inprocess` to `GraphIteratorProto` (defaulting to false) that is set only by the `initialize(std::shared_ptr<ModelProto>)` in-process ORT code path. In `extract_tensor_external_data`, gate the `ORT_MEM_ADDR` branch on this flag: `if (ext_location == detail::ORT_MEM_ADDR && graph_iterator->is_ort_inprocess()) { ... }`. If the flag is false (file-load path), treat the ORT_MEM_ADDR location as an error and throw, e.g. `throw std::runtime_error("ORT_MEM_ADDR sentinel not allowed for file-loaded models");`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #299.
