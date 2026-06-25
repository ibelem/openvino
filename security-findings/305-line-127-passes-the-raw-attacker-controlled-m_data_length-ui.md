# Security finding #305: Line 127 passes the raw attacker-controlled `m_data_length` (uint64…

**Summary:** Line 127 passes the raw attacker-controlled `m_data_length` (uint64…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** DoS via uncontrolled memory allocation: an attacker-controlled ONNX model can request an allocation up to 2^64-1 bytes, causing `aligned_alloc`/`malloc` to fail with `std::bad_alloc` or return nullptr. If AlignedBuffer's allocator does not handle a null return, subsequent `get_ptr<char>()` will dereference null. At minimum, model loading will abort; at worst, a null pointer write occurs.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file `length` field → `m_data_length` (uint64_t) → AlignedBuffer constructor

## Description / Root cause
Line 127 passes the raw attacker-controlled `m_data_length` (uint64_t, set from `std::stoull(entry.value())` at constructor line 26) directly to `ov::AlignedBuffer(m_data_length)`. The AlignedBuffer constructor (aligned_buffer.cpp:18-20) immediately calls `util::aligned_alloc(m_byte_size, alignment)` with no upper-bound check. There is no guard in `load_external_mem_data` that caps `m_data_length` to any sane limit (unlike `load_external_data` which bounds it against actual file size at lines 83-84).

**Validator analysis:** CWE-789 is accurate for the openvino code: load_external_mem_data (line 127) hands the raw, attacker-controlled m_data_length to ov::AlignedBuffer, whose ctor (aligned_buffer.cpp:18-20) does util::aligned_alloc(max(1,byte_size),alignment) with no upper bound and no null check. load_external_data (lines 83-84) caps length against file size, but load_external_mem_data has no equivalent guard. The path is reachable from the ONNX frontend's model-file trust boundary: Tensor::get_external_data (tensor.hpp:324-325) routes to load_external_mem_data whenever data_location()==ORT_MEM_ADDR, and the TensorProto ctor (lines 19-30) reads location/offset/length verbatim from the model, so a crafted .onnx with location='*/_ORT_MEM_ADDR_/*', offset=1, length=0x7FFFFFFFFFFFFFFF reaches line 127 → bad_alloc/null. The impact ('DoS / possible null deref') is correct but understated: this same path also does std::memcpy(dst, reinterpret_cast<char*>(m_offset), m_data_length) at line 129, i.e. it dereferences m_offset as a raw pointer — an arbitrary-read/arbitrary-address primitive that is far worse than the allocation DoS. The proposed fix (upper-bound m_data_length) mitigates the allocation DoS but is insufficient: (1) the correct bound is the size of the actually-registered shared-memory region, which is not currently tracked and would have to be plumbed in; (2) the deeper flaw is that load_external_mem_data is reachable at all from a file-loaded model — the ORT_MEM_ADDR/raw-pointer path should be gated so it is only taken when the EP has explicitly registered the address, never for location strings parsed out of an untrusted model file. AlignedBuffer should also reject byte_size == 0/null-alloc rather than silently producing a buffer whose get_ptr is null. For the EP repo the value is not attacker-amplifiable, so it is rejected.

## Exploit / Proof of Concept
Provide an ONNX model with `location = "*/_ORT_MEM_ADDR_/*"`, `offset = 1` (non-zero to pass `is_valid_buffer` check), `length = 0x7FFFFFFFFFFFFFFF`. Lines 121-124 pass (both non-zero). Line 127 calls `AlignedBuffer(0x7FFFFFFFFFFFFFFF)`. The allocator is asked for ~8 EB; it either throws `std::bad_alloc` (crash/DoS) or returns null, after which line 129 calls `memcpy` on a null source pointer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127
//   (TensorExternalData::load_external_mem_data)
// Pre-fix: a model whose external_data sets location='*/_ORT_MEM_ADDR_/*',
//   offset=1, length=0x7FFFFFFFFFFFFFFF reaches line 127 and calls
//   std::make_shared<ov::AlignedBuffer>(0x7FFFFFFFFFFFFFFF) -> aligned_alloc of
//   ~8EB (aligned_buffer.cpp:18-20, no bound / no null check), then memcpy from
//   reinterpret_cast<char*>(1) at line 129. ASan/the allocator aborts or a null
//   deref occurs.
// Post-fix: load_external_mem_data must reject the oversized / raw-pointer
//   length and the frontend must throw ov::Exception during convert_model.
//
// Harness: ov_onnx_frontend_tests (gtest+ASan), style of onnx_import.in.cpp.
//
// SKELETON: triggering this requires a crafted .onnx fixture (the ORT_MEM_ADDR
// marker + giant length cannot be produced with the in-source model builders
// used elsewhere in onnx_import.in.cpp without a binary fixture), so the model
// file is a TODO.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/external_data/ort_mem_addr_excessive_length.onnx with a
//       TensorProto that has:
//         data_location = EXTERNAL
//         external_data[location] = "*/_ORT_MEM_ADDR_/*"
//         external_data[offset]   = "1"
//         external_data[length]   = "9223372036854775807"  // 0x7FFFFFFFFFFFFFFF
//       and reference it as a Constant/initializer.
OPENVINO_TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_excessive_length.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter=*onnx_external_mem_data_excessive_length_is_rejected*  (substitute the real ${BACKEND_NAME} expansion used by onnx_import.in.cpp). Pre-fix expectation: process aborts under ASan with 'allocation-size-too-big'/'out-of-memory' in util::aligned_alloc (aligned_buffer.cpp:19) or SEGV in the std::memcpy at tensor_external_data.cpp:129 — the EXPECT_THROW does not complete. Post-fix expectation: convert_model throws ov::frontend::onnx::error::invalid_external_data (an ov::Exception) and the test passes. TODO: add the crafted .onnx fixture before this compiles/runs.

## Suggested fix
Add an upper-bound check on `m_data_length` before the allocation, analogous to the file-size guard in `load_external_data` (lines 83-84). Since `load_external_mem_data` operates on a shared-memory region, the valid size should be bounded by the known size of that region. At minimum, add: `if (m_data_length > some_reasonable_max) throw error::invalid_external_data{*this};` before line 127, where `some_reasonable_max` could be a compile-time constant (e.g., 4 GB) or, better, the known size of the registered shared-memory region.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #305.
