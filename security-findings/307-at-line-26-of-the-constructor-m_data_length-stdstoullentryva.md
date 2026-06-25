# Security finding #307: At line 26 of the constructor, `m_data_length = std::stoull(entry.v…

**Summary:** At line 26 of the constructor, `m_data_length = std::stoull(entry.v…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Denial-of-service: an attacker can force the process to attempt an allocation of up to ~16 EB, either exhausting virtual memory and crashing the process or triggering OOM-kill. Combined with the arbitrary-read primitive in finding 1, the length value is also used to control how many bytes are exfiltrated.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted ONNX model file (TensorProto.external_data["length"]) deserialized into TensorExternalData::m_data_length via std::stoull

## Description / Root cause
At line 26 of the constructor, `m_data_length = std::stoull(entry.value())` assigns a fully attacker-controlled value. At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` allocates exactly that many bytes with no upper-bound check. A value of e.g. 0xFFFFFFFFFFFFFF00 will either throw `std::bad_alloc` (DoS) or, on systems where `new` does not throw, return a null/undersized buffer that is then passed to `memcpy` at line 129.

**Validator analysis:** The defect is real for the openvino onnx frontend: load_external_mem_data() (reached from Tensor::get_external_data() when data_location()==ORT_MEM_ADDR, tensor.hpp:324-325) allocates m_data_length bytes with no sanity bound, whereas the sibling loaders (lines 53-54, 83-84) bound the length against the backing file size. A crafted .onnx whose external_data sets location="*/_ORT_MEM_ADDR_/*", a nonzero offset, and length="18446744073709551600" passes the is_valid_buffer check (line 121) and triggers an enormous AlignedBuffer allocation -> std::bad_alloc/OOM. The CWE-789 categorization is accurate, though the more severe consequence (a moderate-but-large length that succeeds, then memcpy at line 129 reads from the attacker-controlled pointer m_offset) overlaps with the companion arbitrary-read finding. The proposed fix (an upper-bound MAX_ALLOWED_TENSOR_BYTES check before line 127, throwing error::invalid_external_data) is correct and sufficient to stop the excessive allocation; cross-checking against the declared shape would be even better since the ORT_MEM_ADDR path has no file size to validate against. For the EP repo the same model could be submitted, but ORT populates the length with the genuine loaded tensor size rather than an arbitrary string, so the excessive-size value is not attacker-controllable from that boundary.

## Exploit / Proof of Concept
Same crafted ONNX model as finding 1. Set "length" to a string such as "18446744073709551600" (UINT64_MAX - 15). The guard at line 121 only checks `m_data_length != 0`, so the check passes. `std::make_shared<ov::AlignedBuffer>(18446744073709551600ULL)` triggers an enormous allocation attempt, crashing any service that loads this model.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-789 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   auto aligned_memory = std::make_shared<ov::AlignedBuffer>(m_data_length);
// where m_data_length is parsed unchecked from TensorProto.external_data["length"]
// (constructor line 26). The model below sets the external_data location to the
// ORT_MEM_ADDR marker, a nonzero offset, and an enormous length so that
// load_external_mem_data() attempts a ~16 EB allocation.
//
// Pre-fix: the unbounded std::make_shared<ov::AlignedBuffer>(m_data_length) throws
//   std::bad_alloc (DoS) instead of a controlled frontend error.
// Post-fix: an upper-bound/shape check rejects the length and convert_model throws
//   ov::Exception (error::invalid_external_data), which EXPECT_THROW captures.
//
// NOTE: this needs a crafted fixture model "external_data_ort_mem_addr_huge_length.onnx"
// ( deterministic to author by hand: a single Constant/initializer with
//  data_location=EXTERNAL and external_data entries
//  location="*/_ORT_MEM_ADDR_/*", offset="4096", length="18446744073709551600").
// Because that binary fixture and the in-memory pointer semantics are not portable,
// this is emitted as a SKELETON.

#include "onnx_utils.hpp"   // TODO: confirm helper header name in src/frontends/onnx/tests/
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: place crafted fixture under onnx/frontend/tests/models/ and name it here.
static const std::string kModel = "external_data_ort_mem_addr_huge_length.onnx";

TEST(ONNXImportExternalData, rejects_excessive_ort_mem_addr_length) {
    // TODO: convert_model() helper signature mirrors onnx_import.in.cpp; verify name.
    EXPECT_THROW(convert_model(kModel), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=ONNXImportExternalData.rejects_excessive_ort_mem_addr_length . Without the fix, the AlignedBuffer allocation at tensor_external_data.cpp:127 throws std::bad_alloc (or AddressSanitizer/allocator reports an allocation-size-too-big abort) escaping as a non-ov::Exception, failing EXPECT_THROW; with a MAX_ALLOWED_TENSOR_BYTES guard the path throws error::invalid_external_data (ov::Exception) and the test passes. TODO: author the crafted .onnx fixture (location="*/_ORT_MEM_ADDR_/*", offset="4096", length="18446744073709551600").

## Suggested fix
Add a maximum-size sanity check before the allocation at line 127: `if (m_data_length > MAX_ALLOWED_TENSOR_BYTES) throw error::invalid_external_data{*this};` where `MAX_ALLOWED_TENSOR_BYTES` is a deployment-appropriate constant (e.g. 2 GiB). Alternatively, cross-check `m_data_length` against the declared tensor shape dimensions to ensure it is consistent with the model's own metadata.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #307.
