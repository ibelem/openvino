# Security finding #384: The istream overload of `process_model` contains the identical unsi…

**Summary:** The istream overload of `process_model` contains the identical unsi…

**CWE IDs:** CWE-191: Integer Underflow / CWE-400: Uncontrolled Resource Consumption
**Severity / Impact:** Uncontrolled memory allocation: `std::string::resize(SIZE_MAX - 47)` or constructing a `Tensor` of shape `{SIZE_MAX - X}` throws `std::bad_alloc`, crashing the caller (DoS). If allocators on a target platform happen to succeed partially, a subsequent `model_stream.read(…, SIZE_MAX - 47)` at line 233 would attempt to read a similarly enormous chunk, causing OOB reads from the stream buffer.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:219` — `ModelDeserializer::process_model(std::istream&)()`
**Validated for repos:** openvino
**Trust boundary:** attacker-writable filesystem cache blob → import_model → ModelDeserializer::process_model(istream)

## Description / Root cause
The istream overload of `process_model` contains the identical unsigned-subtraction `is_valid_model` check at lines 219-222. The same integer underflow in `hdr.consts_offset - hdr.custom_data_offset` (or `hdr.model_offset - hdr.consts_offset`) can be made to equal the corresponding attacker-supplied size field, passing the guard. Afterwards, line 232 calls `xmlInOutString.resize(hdr.custom_data_size)` with the wrapped `SIZE_MAX - 47` value, or line 240 allocates `ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size}))` with a similarly huge value derived from an underflowed `model_offset - consts_offset`.

**Validator analysis:** Confirmed: DataHeader fields are uint64_t/size_t, and the is_valid_model guard at deserializer.cpp:219-221 validates ONLY the consistency equalities, not the orderings (custom_data_offset<=consts_offset<=model_offset). With consts_offset=sizeof(hdr)=48, model_offset=0, the unsigned expression model_offset-consts_offset underflows to SIZE_MAX-47; setting consts_size to that same value satisfies the equality (line 221), custom_data_size==consts_offset-custom_data_offset==0 satisfies line 220, and file_size>model_offset(=0) satisfies the last clause. The guard passes, then line 240 constructs ov::Tensor(element::u8, Shape({SIZE_MAX-47})) which attempts a ~SIZE_MAX-byte allocation → std::bad_alloc (DoS). The same underflow can drive xmlInOutString.resize(hdr.custom_data_size) at line 232. The vulnType (CWE-191 underflow → CWE-400 uncontrolled allocation) and the DoS impact are accurate; the secondary OOB-read claim is weak because allocation almost always fails first, but the DoS is solid. The proposed fix is correct and sufficient: add explicit ordering assertions (hdr.custom_data_offset<=hdr.consts_offset<=hdr.model_offset<=file_size) before the equality checks to eliminate the underflow, plus upper-bound checks (consts_size<=file_size, custom_data_size<=file_size) before the resize/Tensor allocations. This mirrors the needed fix in the mmap overload at lines 157-159, which shares the identical flaw.

## Exploit / Proof of Concept
Same crafted header as Finding 1 but delivered via the istream code path (e.g., direct `import_model` with a file stream). Alternatively, set `consts_offset = sizeof(hdr) = 48`, `model_offset = 0`, `consts_size = SIZE_MAX - 47` (from `0 - 48`), `custom_data_size = 0`, `custom_data_offset = 48`, and `file_size > 0`. The `is_valid_model` check passes, and line 240 tries to construct `ov::Tensor(u8, Shape({SIZE_MAX - 47}))`, throwing `std::bad_alloc` (DoS).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for openvino intel_cpu deserializer.cpp:219-243 (process_model istream overload).
// Pre-fix: a crafted StreamSerialize::DataHeader with consts_offset=sizeof(hdr), model_offset=0
// underflows (model_offset - consts_offset) to ~SIZE_MAX, passes is_valid_model (line 222),
// then ov::Tensor(u8, Shape({hdr.consts_size})) at line 240 attempts a ~SIZE_MAX allocation -> std::bad_alloc.
// Post-fix: header ordering/upper-bound validation must reject the blob with ov::Exception before allocation.
//
// Target: ov_cpu_unit_tests (component test tree under src/plugins/intel_cpu/tests/unit/).
// NOTE: ModelDeserializer::process_model is private; the realistic entry point is
// ov::Core::import_model(stream, "CPU") with a cache blob, OR exercising ModelDeserializer directly.
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
#include "openvino/runtime/core.hpp"

namespace {
// TODO: replace with the real layout of ov::pass::StreamSerialize::DataHeader
// (read src/core/.../pass/serialize/stream_serialize.hpp to confirm exact field order/types).
struct DataHeaderMirror {
    uint64_t custom_data_offset;
    uint64_t custom_data_size;
    uint64_t consts_offset;
    uint64_t consts_size;
    uint64_t model_offset;
    uint64_t model_size;
};

TEST(CpuModelDeserializer, UnderflowedConstsSizeIsRejected) {
    DataHeaderMirror hdr{};
    hdr.custom_data_offset = sizeof(DataHeaderMirror);   // must equal sizeof(hdr)
    hdr.custom_data_size   = 0;                          // == consts_offset - custom_data_offset
    hdr.consts_offset      = sizeof(DataHeaderMirror);
    hdr.model_offset       = 0;                          // forces underflow below
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset; // wraps to ~SIZE_MAX, == model_offset-consts_offset
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.append(64, '\0'); // ensure file_size > model_offset(=0)
    std::stringstream ss(blob);

    // TODO: confirm import_model signature/device for the cache-deserialize path that lands in
    // ModelDeserializer::process_model(istream). Pre-fix this throws std::bad_alloc at deserializer.cpp:240;
    // post-fix it must throw ov::Exception from the validated header guard instead of allocating.
    ov::Core core;
    EXPECT_THROW(core.import_model(ss, "CPU"), ov::Exception);
}
} // namespace
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter=CpuModelDeserializer.UnderflowedConstsSizeIsRejected. Pre-fix expectation (with ASan): allocator abort / std::bad_alloc (or ASan allocation-size-too-big) at deserializer.cpp:240 when constructing ov::Tensor(u8, Shape({~SIZE_MAX})); post-fix expectation: ov::Exception thrown from the header validation guard (lines 219-222) before any allocation, satisfying EXPECT_THROW. TODO: confirm DataHeader layout and the correct import_model/ModelDeserializer entry point by reading the intel_cpu test tree and stream_serialize.hpp.

## Suggested fix
Apply the same fix as Finding 1: add explicit `>=` ordering assertions before the equality checks to prevent unsigned underflow. Additionally, add an upper-bound check — `hdr.custom_data_size <= file_size` and `hdr.consts_size <= file_size` — before the `resize` and `Tensor` allocation calls at lines 232 and 240, rejecting any header whose size fields exceed the actual stream size.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #384.
