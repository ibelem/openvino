# Security finding #125: m_data_length is parsed from the untrusted ONNX 'length' key by std…

**Summary:** m_data_length is parsed from the untrusted ONNX 'length' key by std…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Uncontrolled resource consumption / DoS: a single malicious ONNX model causes the inference host to attempt an enormous heap allocation, triggering std::bad_alloc or OS-level OOM (process kill, system memory pressure). In container or embedded deployments this can kill co-located workloads.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file on disk → TensorProto::external_data 'length' field → TensorExternalData constructor → load_external_mem_data()

## Description / Root cause
m_data_length is parsed from the untrusted ONNX 'length' key by std::stoull at constructor line 26 with no upper-bound check. At line 127 it is passed directly to std::make_shared<ov::AlignedBuffer>(m_data_length). std::stoull accepts values up to 2^64-1, so an attacker can request an allocation of up to 16 exabytes. Unlike load_external_data() and load_external_mmap_data() (which bound m_data_length against the actual file_size), load_external_mem_data() has no corresponding bound.

**Validator analysis:** CWE-789 is accurate: m_data_length comes from the attacker-controlled external_data 'length' value (std::stoull, line 26) and is allocated unchecked at line 127, while the sibling paths load_external_data (line 83-84) and load_external_mmap_data (line 53-54) both bound it against the real file_size; load_external_mem_data has no backing file and therefore no bound at all. The path is reachable from the openvino ONNX frontend's trust boundary (convert_model on a crafted .onnx) because has_external_data()/get_external_data() (tensor.hpp:308-332) dispatch to load_external_mem_data whenever location equals the ORT_MEM_ADDR marker string, which a model file can set literally. The DoS/impact claim is correct (bad_alloc for absurd sizes, real OOM/commit for plausible multi-GB sizes). One caveat the finding under-states: this same path also reinterpret_casts the attacker-controlled m_offset as a raw pointer (line 126) and memcpy's from it (line 129), i.e. an arbitrary-read/crash that is arguably more severe than the allocation DoS — but the allocation at line 127 executes first, so the CWE-789 trigger is valid. The proposed fix (bound m_data_length by the parent TensorProto shape*element_size before allocation, throwing error::invalid_external_data) is correct and sufficient for the allocation defect; it would be stronger to additionally reject/validate the offset-as-pointer trust assumption on this path, but that is out of scope for CWE-789.

## Exploit / Proof of Concept
Set external_data {key:'location', value:'*/_ORT_MEM_ADDR_/*'}, {key:'offset', value:'4096'}, {key:'length', value:'18446744073709551615'} (UINT64_MAX). load_external_mem_data() passes the guard (offset!=0), then at line 127 requests a ~16 EB AlignedBuffer, causing bad_alloc or memory exhaustion before the memcpy at line 129 even runs.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in:
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   (TensorExternalData::load_external_mem_data) — m_data_length parsed unchecked
//   at line 26 and allocated unbounded at line 127.
//
// Encodes the fix: a model whose external_data declares location == ORT_MEM_ADDR
// and an enormous 'length' must be REJECTED (ov::Exception / invalid_external_data)
// rather than attempting a ~16 EB AlignedBuffer allocation.
//
// Pre-fix: ASan/allocator reports the huge allocation request (or std::bad_alloc),
//          OR the arbitrary-pointer memcpy crashes — test does NOT throw ov::Exception cleanly.
// Post-fix: convert_model throws ov::Exception due to the upper-bound check.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// SKELETON: requires a crafted .onnx fixture that cannot be generated read-only.
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // FrontEndTestUtils / convert_model helper

using namespace ov::frontend::onnx::tests;

// TODO(fixture): create models/external_data/excessive_mem_length.onnx that contains
//   a TensorProto with data_location = EXTERNAL and external_data entries:
//     {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
//     {key:"offset",   value:"4096"}
//     {key:"length",   value:"18446744073709551615"}   // UINT64_MAX
//   The declared tensor shape/element type must imply a far smaller byte size,
//   so the fix's shape*elt_size bound rejects the oversized 'length'.
//   (A pure-text .onnx cannot be emitted from a read-only tree; hand-build the
//    protobuf with onnx.helper or check in the binary fixture.)
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    EXPECT_THROW(convert_model("external_data/excessive_mem_length.onnx"), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_onnx_frontend_tests --gtest_filter='*onnx_external_mem_data_excessive_length_is_rejected*'. Expected pre-fix: ASan 'allocation-size-too-big' / 'requested allocation size exceeds maximum supported size' (or std::bad_alloc) at tensor_external_data.cpp:127, test fails (no ov::Exception). Expected post-fix: convert_model throws ov::Exception (invalid_external_data) and the test passes.

## Suggested fix
Add an explicit upper-bound check before the AlignedBuffer allocation: e.g. 'if (m_data_length > MAX_ALLOWED_TENSOR_BYTES) throw error::invalid_external_data{*this};'. A reasonable limit is the product of the declared tensor shape and element size (computable from the parent TensorProto), enforced in TensorExternalData constructor or at the call site in get_ov_constant().


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #125.
