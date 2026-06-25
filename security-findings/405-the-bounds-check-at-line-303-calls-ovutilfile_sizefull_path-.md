# Security finding #405: The bounds check at line 303 calls `ov::util::file_size(full_path)`…

**Summary:** The bounds check at line 303 calls `ov::util::file_size(full_path)`…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition / CWE-125: Out-of-bounds Read
**Severity / Impact:** An attacker who controls both the ONNX model file (or its external_data 'offset' field) and the writable external data file can trigger an out-of-bounds read from memory pages beyond the mmap window. The resulting pointer is stored in `tensor_meta_info.m_tensor_data` and later used by the inference engine to read tensor weights, causing either a process crash (SIGSEGV on unmapped page) or silent exposure of adjacent process memory as model weights. Severity is high for environments where untrusted ONNX models are loaded.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:303` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX TensorProto.external_data[].offset and a writable external data file path

## Description / Root cause
The bounds check at line 303 calls `ov::util::file_size(full_path)` (a stat-based snapshot) and validates `ext_data_offset` + `ext_data_length` against it. Then at line 323 `ov::load_mmap_object(full_path)` independently maps the file. There is no post-mmap re-validation: the pointer arithmetic at line 328 — `mapped_memory->data() + ext_data_offset` — uses the protobuf-supplied `ext_data_offset` directly against the mapped region without checking `ext_data_offset < mapped_memory->size()` or `ext_data_offset + resolved_data_length <= mapped_memory->size()`.

**Validator analysis:** The static bounds check at lines 304-305 is itself sound (no integer underflow: the `ext_data_length > file_size` term short-circuits before the subtraction), so in the absence of concurrent file modification there is NO bug — the whole file is mmap'd and mapped_memory->size() == file_size, making data()+offset in-bounds. The only real defect is the genuine TOCTOU window between the stat at line 303 and the mmap at line 323: a writable/symlink-swappable external-data file can be truncated in that window, after which data()+ext_data_offset points past the (now smaller) mapping. CWE-367 is accurate; CWE-125 is the consequence. However the stated impact is overstated: accessing a truncated mmap beyond EOF produces SIGBUS/SIGSEGV (DoS), not 'silent exposure of adjacent process memory' — mmap is a file mapping, not arbitrary heap, so cross-region info disclosure is not the realistic outcome. The proposed fix is essentially correct and placed correctly (after both the cache-hit and fresh-map branches), but should re-derive resolved_data_length AFTER the post-mmap size is known and guard mapped_memory->data()==nullptr; better: validate `mapped_memory && ext_data_offset <= mapped_memory->size() && (mapped_memory->size() - ext_data_offset) >= resolved_data_length` before line 327, throwing FRONT_END_GENERAL_CHECK rather than std::runtime_error to match the frontend's error convention. For the EP repo the path is not reachable because OVEP routes self-managed weights through ORT_MEM_ADDR (early return at line 297).

## Exploit / Proof of Concept
1. Craft an ONNX model with `external_data[].location = 'weights.bin'` and `offset = 65536`, `length = 1024` where `weights.bin` is initially ≥ 67560 bytes — this passes the `file_size()` check at line 303. 2. Race: between line 303 and line 323, truncate `weights.bin` to 0 bytes (e.g. via a concurrent writer or symlink swap). 3. `load_mmap_object()` maps a 0-byte file; `mapped_memory->size()` is 0 but `ext_data_offset = 65536`. 4. Line 328 computes `mapped_memory->data() + 65536` — an OOB pointer into unmapped address space, which is subsequently dereferenced by the runtime when consuming tensor data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-367 TOCTOU at
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:303-328
// Pre-fix: file_size() snapshot at :303 validates offset/length, but the mmap at :323
// is never re-validated, so data()+ext_data_offset at :328 can be out of bounds when the
// external-data file is smaller than the stat-time size (truncation/symlink race).
// This test asserts that loading a model whose external_data offset+length exceeds the
// ACTUAL mapped file size is rejected (throws) instead of producing an OOB pointer.
//
// NOTE (skeleton): a pure C++ unit test cannot reproduce the live race window between
// stat and mmap. Instead it must stage a crafted model + external file where the mapped
// size is smaller than offset+length so the post-mmap check fires deterministically.
// TODO: provide binary fixtures (see TODOs) — exact bytes are not derivable by reading source.

#include "onnx_utils.hpp"            // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/file_utils.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: create a temp dir containing:
//   - external_data_truncated.onnx : a TensorProto with external_data
//       location="weights.bin", offset="65536", length="1024"
//   - weights.bin : a file SMALLER than offset+length (e.g. 16 bytes) so the mmap is
//                   smaller than the protobuf-claimed range. (To exercise the static
//                   check instead, this also covers offset beyond file end.)
// The pre-fix code dereferences mapped_memory->data()+65536 -> ASan/SIGSEGV/SIGBUS.
// The post-fix code must throw before constructing m_tensor_data.
TEST(onnx_external_data, mmap_offset_beyond_actual_file_size_is_rejected) {
    const std::string model_path =
        ov::test::utils::getModelFromTestModelZoo(
            std::string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_truncated.onnx");
    // Expect a throw rather than an OOB read when offset/length exceed the mapped size.
    EXPECT_THROW(convert_model(model_path), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_onnx_frontend_tests --gtest_filter=onnx_external_data.mmap_offset_beyond_actual_file_size_is_rejected . Pre-fix expected failure: AddressSanitizer/SEGV or SIGBUS at graph_iterator_proto.cpp:328 (mapped_memory->data()+ext_data_offset out of bounds), or the test fails because no exception is thrown. Post-fix expected: EXPECT_THROW passes (ov::Exception thrown by the new post-mmap bounds check). TODO: add the external_data_truncated.onnx + undersized weights.bin fixtures under the ONNX test models dir before running.

## Suggested fix
After `load_mmap_object()` succeeds (line 323), add a post-mmap bounds check before the pointer assignment: `if (mapped_memory == nullptr || ext_data_offset > mapped_memory->size() || (mapped_memory->size() - ext_data_offset) < resolved_data_length) { throw std::runtime_error("External data file changed between validation and mapping"); }`. This closes the TOCTOU window by re-validating the offset against the actual mapped size rather than the earlier stat result.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #405.
