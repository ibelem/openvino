# Security finding #382: At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` a…

**Summary:** At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` a…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Uncontrolled heap allocation / out-of-memory DoS. An attacker supplying a crafted ONNX model with a very large `length` value causes the process to attempt an enormous allocation, resulting in `std::bad_alloc` or OOM-kill, crashing the inference service.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** ONNX model's `external_data.length` string field parsed via `std::stoull` into `uint64_t m_data_length` at constructor line 26

## Description / Root cause
At line 127, `std::make_shared<ov::AlignedBuffer>(m_data_length)` allocates a buffer whose size comes directly from the attacker-supplied `length` field, with no upper-bound check. `m_data_length` is a `uint64_t` that can be up to 2^64-1; the only guard (line 121-123) only checks non-zero. A value such as `0x7fffffffffffffff` will be passed directly to the allocator.

**Validator analysis:** Confirmed real and reachable. has_external_data() (tensor.hpp:312-313) is true when data_location==EXTERNAL; get_external_data() (tensor.hpp:317-332) constructs TensorExternalData(*m_tensor_proto) and, when the external_data 'location' string equals the marker ORT_MEM_ADDR ('*/_ORT_MEM_ADDR_/*'), dispatches to load_external_mem_data(). There, the only guard is is_valid_buffer = m_offset && m_data_length (line 121); offset=1,length=0x7FFFFFFFFFFFFFFF passes it, and line 127 calls std::make_shared<ov::AlignedBuffer>(m_data_length). AlignedBuffer's ctor (aligned_buffer.cpp:18-20) calls util::aligned_alloc(m_byte_size) with no cap -> bad_alloc/OOM DoS. CWE-789 and the DoS impact are accurate. NOTE: the finding understates severity — line 126-129 reinterpret_cast m_offset to a raw pointer and memcpy m_data_length bytes from it, i.e. an attacker-controlled arbitrary-address read (CWE-125), which is worse than the allocation DoS; this whole ORT_MEM_ADDR path trusts offset as a process address and should never be entered for a disk-parsed model. The proposed fix (upper-bound check on m_data_length before line 127) mitigates the allocation DoS but does NOT fix the arbitrary-pointer memcpy. A stronger fix: refuse the ORT_MEM_ADDR location entirely when the tensor originates from a parsed TensorProto (it is only legitimate via the EP's in-memory TensorPlace path), and additionally bound m_data_length. Both repos validated: openvino holds the defect; the EP is the untrusted-model entry boundary that reaches it.

## Exploit / Proof of Concept
Set `offset = 1` and `length = 9223372036854775807` (0x7FFFFFFFFFFFFFFF) in the external_data entries with `location = "*/_ORT_MEM_ADDR_/*"`. The non-zero check at line 121 passes. Line 127 calls `AlignedBuffer(9223372036854775807)`, exhausting virtual memory and crashing the process.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 / arbitrary-address read at
// targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
// (and the related memcpy at line 129). A crafted ONNX model whose tensor sets
//   data_location = EXTERNAL,
//   external_data{ location = "*/_ORT_MEM_ADDR_/*", offset = "1",
//                  length = "9223372036854775807" }
// must be rejected (invalid_external_data / ov::Exception) instead of attempting
// an enormous AlignedBuffer allocation + memcpy from a bogus address.
//
// Pre-fix: AlignedBuffer(0x7fffffffffffffff) -> bad_alloc/OOM, or (if it returns)
//          memcpy from addr 0x1 -> ASan/segv. Post-fix: convert_model throws.
//
// Place in: src/frontends/onnx/tests/onnx_import.in.cpp
// TODO: provide the crafted fixture models/ort_mem_addr_huge_length.onnx that
//       carries the EXTERNAL data_location + ORT_MEM_ADDR location with the
//       oversized length string (cannot be generated as a self-contained
//       string literal here — needs a serialized TensorProto).
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_data_ort_mem_addr_huge_length_rejected) {
    EXPECT_THROW(convert_model("ort_mem_addr_huge_length.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_data_ort_mem_addr_huge_length_rejected*. Expected pre-fix: AddressSanitizer 'allocation-size-too-big' / std::bad_alloc abort, or SEGV/ASan READ on memcpy from address 0x1 at tensor_external_data.cpp:129; expected post-fix: convert_model throws ov::Exception (invalid_external_data) and the test passes. TODO: add the crafted ort_mem_addr_huge_length.onnx fixture under the onnx test models dir.

## Suggested fix
Add an explicit upper-bound check on `m_data_length` before allocation, e.g.: `if (m_data_length > MAX_TENSOR_BYTES) throw error::invalid_external_data{*this};` where `MAX_TENSOR_BYTES` is a reasonable constant (e.g., 2 GB). This is consistent with the bounds checking already present in `load_external_data()` and `load_external_mmap_data()` (lines 53-55) which validate length against the actual file size.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #382.
