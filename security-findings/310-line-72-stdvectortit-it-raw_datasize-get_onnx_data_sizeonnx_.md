# Security finding #310: Line 72: `std::vector<T>(it, it + (raw_data.size() / get_onnx_data_…

**Summary:** Line 72: `std::vector<T>(it, it + (raw_data.size() / get_onnx_data_…

**CWE IDs:** CWE-125: Out-of-bounds Read (CWE-843: Type Confusion)
**Severity / Impact:** Heap out-of-bounds read: an attacker-controlled ONNX model can force the library to read N×K bytes past the end of a protobuf raw_data string buffer (where K = sizeof(T)/get_onnx_data_size(data_type), up to 8× for double/int64 vs. 1-byte ONNX types). The over-read data is copied into a returned std::vector<T>, potentially exposing heap metadata or adjacent allocation contents (info-leak). If raw_data is small and the heap layout is adversarial, this can also cause a segfault/crash (DoS). Affects any process that calls ov::frontend::onnx to load a crafted .onnx file.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:72` — `detail::__get_raw_data<T>()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model file → ov::frontend::onnx ONNX frontend tensor parsing

## Description / Root cause
Line 72: `std::vector<T>(it, it + (raw_data.size() / get_onnx_data_size(onnx_data_type)))` uses `onnx_data_type` to compute element_count but uses `sizeof(T)` (implicitly, via the iterator) to advance the pointer. When `sizeof(T) > get_onnx_data_size(onnx_data_type)`, the vector reads `element_count * sizeof(T)` bytes from a heap buffer of only `raw_data.size() = element_count * get_onnx_data_size(onnx_data_type)` bytes. No assertion or check validates that the two sizes agree. Every call site in tensor.cpp (lines 61, 81, 114, 145, 166, 188, 209, 230, 251, 273, 294, 315, 338, 371, 403) passes `m_tensor_proto->data_type()` directly without verifying it matches the template type T. The raw-data branch precedes the data_type equality guard, so any tensor with `has_raw_data()==true` bypasses the type check entirely.

**Validator analysis:** The defect is real. __get_raw_data<T> computes the element count from the ONNX type's byte size (get_onnx_data_size(onnx_data_type)) but advances the const T* iterator by sizeof(T); when sizeof(T) exceeds the ONNX element size the std::vector<T> range [it, it+count) spans count*sizeof(T) > raw_data.size() bytes, an out-of-bounds heap read whose contents are copied into the returned vector (CWE-125, with the underlying CWE-843 type confusion). The raw-data branch in every get_data<T>() specialization (tensor.cpp:61,81,114,145,166,188,209,230,251,273,294,315,338,371,403) runs before any data_type==T guard, so a model with has_raw_data()==true and a mismatched data_type bypasses type checking. The Constant-import path (get_ov_constant) is type-matched and self-safe, but direct hardcoded-T call sites such as get_absolute_indices's indices_tensor.get_data<int64_t>() (constant.cpp:86,182) take the data_type from the model and are reachable with a smaller declared type plus raw_data, confirming reachability from a crafted .onnx. The vulnType and info-leak/DoS impact are accurate. The proposed fix direction (validate before building the vector) is right, but the exact assertion `sizeof(T) == get_onnx_data_size(onnx_data_type)` is too strict and would wrongly reject legitimate nibble types (INT4/UINT4 read via get_data<int8_t>/<uint8_t>, where the ONNX element size is sub-byte). A cleaner, minimally-invasive fix is to advance using sizeof(T) consistently and require alignment, mirroring raw_value_count(): `OPENVINO_ASSERT(raw_data.size() % sizeof(T) == 0); return std::vector<T>(it, it + raw_data.size()/sizeof(T));` — this guarantees the read stays in bounds. Better still, also assert the declared data_type matches the expected type set per specialization to prevent silent type confusion of the bytes.

## Exploit / Proof of Concept
Craft an ONNX model with a Constant tensor whose `data_type = FLOAT8E4M3FN` (get_onnx_data_size = 1) and `raw_data` = 8 bytes. When the frontend calls `get_data<double>()` on this tensor (which occurs if any op or downstream cast requests double values from this node), the function computes element_count = 8/1 = 8, then constructs std::vector<double>(it, it+8), reading 64 bytes from a 8-byte heap buffer. The extra 56 bytes are read from adjacent heap memory, which may contain pointers or sensitive data. Same mismatch applies to get_data<int64_t> / get_data<uint64_t> (8× over-read), get_data<float>/get_data<int32_t> (4× over-read with a 1-byte ONNX type), and get_data<ov::float16>/get_data<int16_t> (2× over-read).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in ov::frontend::onnx detail::__get_raw_data<T>
// (targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:72).
// The flaw: element_count is derived from get_onnx_data_size(onnx_data_type)
// but the const T* iterator advances by sizeof(T); a raw tensor declared with a
// 1-byte ONNX type (e.g. INT8/UINT8/FLOAT8E4M3FN) but consumed via get_data<int64_t>()
// reads count*8 bytes from a count-byte raw_data buffer -> heap-buffer-overflow.
//
// This encodes the fix as: loading a crafted .onnx whose initializer/sparse-indices
// tensor has has_raw_data()==true and a data_type whose byte size is smaller than the
// type the importer reads it as must be REJECTED with ov::Exception (not silently OOB-read).
//
// Pre-fix: triggers ASan heap-buffer-overflow inside __get_raw_data.
// Post-fix: convert_model throws ov::Exception due to size/type validation.
//
// TODO: provide the crafted fixture models/onnx/tensor_raw_data_type_mismatch.onnx :
//   - a Constant/initializer tensor (or SparseTensor indices) with raw_data of N bytes,
//     data_type = INT8 (or FLOAT8E4M3FN), but referenced where the frontend calls
//     get_data<int64_t>() (e.g. sparse-tensor indices in opset_13 Constant).
//   The exact onnx must be built with the onnx python helper; binary fixture cannot be
//   authored inline here.

#include "onnx_utils.hpp"  // TODO: confirm helper header used by onnx_import.in.cpp (convert_model)
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// static std::string s_manifest = "...";  // TODO: match onnx_import.in.cpp manifest pattern

OPENVINO_TEST(onnx_import_validation, tensor_raw_data_type_size_mismatch_is_rejected) {
    // TODO: place crafted model under the frontend test models dir.
    EXPECT_THROW(convert_model("tensor_raw_data_type_mismatch.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (build with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=*tensor_raw_data_type_size_mismatch_is_rejected*. Pre-fix expectation: ASan reports 'heap-buffer-overflow READ' originating in detail::__get_raw_data (tensor.hpp:72) while constructing std::vector<int64_t> from a raw_data buffer sized for a 1-byte ONNX type. Post-fix expectation: convert_model throws ov::Exception (size/type validation) and the test passes with no ASan report. TODO: supply the crafted .onnx fixture (binary) — a self-contained inline test is not achievable because the trigger requires a malformed protobuf with has_raw_data()==true and a deliberately mismatched data_type.

## Suggested fix
Add a static or runtime assertion inside `__get_raw_data` before constructing the vector. For example:
```cpp
template <typename T>
inline std::vector<T> __get_raw_data(const std::string& raw_data, int onnx_data_type) {
    const size_t onnx_elem_size = get_onnx_data_size(onnx_data_type);
    OPENVINO_ASSERT(sizeof(T) == onnx_elem_size,
        "raw_data type mismatch: sizeof(T)=", sizeof(T),
        " != onnx element size=", onnx_elem_size);
    auto it = reinterpret_cast<const T*>(raw_data.data());
    OPENVINO_ASSERT(raw_data.size() % sizeof(T) == 0,
        "raw_data size not aligned with element size");
    return std::vector<T>(it, it + (raw_data.size() / sizeof(T)));
}
```
Alternatively, each `get_data<T>()` specialisation should assert `data_type == expected_type` before entering the raw-data branch, mirroring the non-raw-data check that already exists in each specialisation.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #310.
