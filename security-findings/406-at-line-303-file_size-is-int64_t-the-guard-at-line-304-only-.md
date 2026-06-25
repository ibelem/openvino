# Security finding #406: At line 303 `file_size` is `int64_t`. The guard at line 304 only th…

**Summary:** At line 303 `file_size` is `int64_t`. The guard at line 304 only th…

**CWE IDs:** CWE-195: Signed to Unsigned Conversion Error / CWE-20: Improper Input Validation
**Severity / Impact:** If the downstream `ov::load_mmap_object` or stream allocator does not independently reject the error condition, the engine proceeds with a `resolved_data_length` near `SIZE_MAX`. Passing this value to memory-allocation or read APIs leads to CWE-789 (excessive allocation) or buffer overflow when the runtime later uses `m_tensor_data_size` to copy or iterate over tensor data.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:304` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Return value of `ov::util::file_size()` (int64_t) cast to uint64_t in the bounds guard; attacker-controlled `ext_data_offset` and `ext_data_length` from ONNX protobuf

## Description / Root cause
At line 303 `file_size` is `int64_t`. The guard at line 304 only throws when `(file_size <= 0 && ext_data_length > 0)`. If `file_size < 0` (stat error) AND `ext_data_length == 0` (omitted in protobuf), condition 1 is false; conditions 2 and 3 cast `file_size` to `uint64_t`, wrapping it to `UINT64_MAX`, so both comparisons evaluate to false — no exception is thrown. Execution falls through to line 314 where `resolved_data_length = static_cast<size_t>(file_size) - static_cast<size_t>(ext_data_offset)` = `SIZE_MAX - ext_data_offset`, a pathologically large length value.

**Validator analysis:** The logic flaw is real and present in openvino core. ov::util::file_size (file_util.hpp:151-155) returns int64_t -1 on any std::filesystem::file_size error. The bounds guard (line 304-305) only rejects file_size<=0 in conjunction with ext_data_length>0; when ext_data_length is omitted (==0), condition 1 collapses to false, and conditions 2/3 cast the negative file_size to uint64_t (UINT64_MAX), so ext_data_length(0)>UINT64_MAX is false and ext_data_offset>UINT64_MAX-0 is false. Execution reaches line 312-314 where the ternary takes the else branch and computes (size_t)file_size - (size_t)ext_data_offset = SIZE_MAX - offset. This resolved_data_length is then stored as m_tensor_data_size and, in External_Stream mode, passed to allocate_data() at line 353 (CWE-789 excessive allocation / bad_alloc), or in External_MMAP mode stored as the tensor size while data points into a possibly-tiny mapping (OOB read) — the latter only under a TOCTOU/FUSE race where file_size errors but mmap succeeds, as the exploit notes. CWE-195/CWE-20 categorization is accurate; the integer-underflow root cause is genuine. Impact is somewhat overstated toward guaranteed buffer overflow — the most reliable outcome is excessive-allocation DoS, with OOB only in the narrow mmap-race case — but the defect stands. The proposed fix is correct and sufficient: making the check `if (file_size <= 0 || ext_data_length > (uint64_t)file_size || ext_data_offset > (uint64_t)file_size - ext_data_length)` unconditionally rejects any non-positive file_size before the casts are reached, eliminating the bypass; alternatively, validate file_size>=0 in a dedicated branch immediately after line 303.

## Exploit / Proof of Concept
Supply an ONNX model whose `external_data` entry has `location` pointing to a file that transiently returns -1 from `stat()` (e.g. a FUSE mount returning EACCES during the stat but readable during mmap), and omit the `length` key (`ext_data_length` defaults to 0). The guard at lines 304-305 passes because `static_cast<uint64_t>(-1) = UINT64_MAX`; `0 > UINT64_MAX` is false and `ext_data_offset > UINT64_MAX` is false. `resolved_data_length` becomes `SIZE_MAX - ext_data_offset`. The engine then attempts to allocate or access `SIZE_MAX` bytes of tensor data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for graph_iterator_proto.cpp:304-314 (CWE-195/CWE-20):
// an ONNX external_data tensor whose backing file yields file_size()<=0 while
// the protobuf omits the "length" key (ext_data_length==0) must be REJECTED
// before resolved_data_length underflows to ~SIZE_MAX.
//
// Pre-fix: guard at line 304 is bypassed (file_size<=0 only checked when
//   ext_data_length>0), resolved_data_length = (size_t)file_size - offset
//   = SIZE_MAX-offset -> allocate_data(SIZE_MAX) / OOB tensor size.
// Post-fix: unconditional file_size<=0 check throws here.
//
// SKELETON: triggering file_size()==-1 needs a backing path that errors on
// std::filesystem::file_size but still satisfies the chosen memory mode, and a
// crafted .onnx that points external_data->location at it with offset set and
// length omitted. Both require fixtures this skeleton cannot embed.

#include <gtest/gtest.h>
#include "openvino/frontend/manager.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::frontend;

TEST(onnx_external_data, external_data_negative_filesize_length_omitted_is_rejected) {
    FrontEndManager fem;
    auto fe = fem.load_by_framework("onnx");
    ASSERT_NE(fe, nullptr);

    // TODO: provide a model whose external_data entry has:
    //   key "location" -> a path for which ov::util::file_size() returns -1
    //                     (e.g. a directory, or a special/unstattable file)
    //   key "offset"   -> some non-zero value
    //   (NO "length" key, so ext_data_length defaults to 0)
    // const std::string model_path = "<crafted_external_data_neg_size>.onnx";
    // auto input_model = fe->load(model_path);
    // ASSERT_NE(input_model, nullptr);

    // The convert MUST throw rather than computing resolved_data_length=SIZE_MAX.
    // EXPECT_THROW(fe->convert(input_model), ov::Exception);

    GTEST_SKIP() << "TODO: supply crafted .onnx fixture + a path that makes "
                    "ov::util::file_size() return -1 while ext_data_length==0";
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests --config Release (build with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data.external_data_negative_filesize_length_omitted_is_rejected . Pre-fix expectation: no exception thrown and downstream allocate_data(~SIZE_MAX) triggers std::bad_alloc / ASan allocation-size-too-big; post-fix: ov::Exception/std::runtime_error('Invalid usage of method for externally stored data') thrown at graph_iterator_proto.cpp:310.

## Suggested fix
Explicitly guard for `file_size < 0` unconditionally: change line 304 to `if (file_size <= 0 || ext_data_length > static_cast<uint64_t>(file_size) || ext_data_offset > static_cast<uint64_t>(file_size) - ext_data_length)`. The single `file_size <= 0` check (without the `&& ext_data_length > 0` conjunction) ensures any negative or zero size immediately rejects the tensor, eliminating the sign-extension bypass for the `ext_data_length == 0` case.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #406.
