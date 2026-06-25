# Security finding #388: The `is_valid_model` validation at lines 157-159 (buffer path) and …

**Summary:** The `is_valid_model` validation at lines 157-159 (buffer path) and …

**CWE IDs:** CWE-191: Integer Underflow (leading to CWE-789: Memory Allocation with Excessive Size Value)
**Severity / Impact:** After bypassing validation, `hdr.custom_data_size = 0xFFFFFFFFFFFFFFD0` (~18 EiB) is used as a size argument: in the buffer path at line 167 it is passed to `pugixml::xml_document::load_buffer(..., hdr.custom_data_size, ...)` causing an OOB read over the entire heap; in the stream path at line 232 `xmlInOutString.resize(hdr.custom_data_size)` triggers `std::bad_alloc` / memory exhaustion (DoS). Either path is reachable from `Plugin::import_model` / `Plugin::import_model(Tensor)` without authentication.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:157` — `ModelDeserializer::process_model(std::shared_ptr<ov::AlignedBuffer>) / ModelDeserializer::process_model(std::istream)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled model cache blob (AlignedBuffer or istream) passed to import_model / ModelDeserializer

## Description / Root cause
The `is_valid_model` validation at lines 157-159 (buffer path) and 219-221 (stream path) evaluates `hdr.consts_offset - hdr.custom_data_offset` using unsigned `size_t` arithmetic without checking that `hdr.consts_offset >= hdr.custom_data_offset` first. If an attacker sets `consts_offset=0` and `custom_data_offset=48` (= `sizeof(DataHeader)` on 64-bit, which satisfies the first condition), the subtraction wraps to `0xFFFFFFFFFFFFFFD0`. The check `hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset` is then satisfied by setting `custom_data_size = 0xFFFFFFFFFFFFFFD0`. Remaining conditions (`consts_size == model_offset - consts_offset`, `file_size > model_offset`) are satisfied by setting `consts_size=0, model_offset=0` in a non-empty file. The blob therefore passes the entire `is_valid_model` guard.

**Validator analysis:** Confirmed real for openvino. DataHeader members are unsigned size_t (serialize.hpp:68-75) and sizeof(DataHeader)=48 on 64-bit. The is_valid_model expression at deserializer.cpp:157-159 (buffer) and 219-221 (stream) checks `custom_data_size == consts_offset - custom_data_offset` without first asserting `consts_offset >= custom_data_offset`. Setting custom_data_offset=48, consts_offset=0, custom_data_size=0xFFFFFFFFFFFFFFD0, consts_size=0, model_offset=0 satisfies all four conditions via wrap-around with a non-empty file (file_size>0). The wrapped value then flows into pugixml load_buffer (:167) or std::string::resize (:232). CWE-191→CWE-789 categorization is accurate. Impact: the stream path resize is the clearest DoS (std::length_error/bad_alloc, uncaught → abort). The buffer-path load_buffer claim of a full-heap OOB read is overstated: pugixml's load_buffer allocates a fresh `size`-byte buffer before memcpy, so an 18-EiB request returns status_out_of_memory or throws bad_alloc rather than reading OOB — still a DoS, not necessarily OOB disclosure. The proposed fix is correct and sufficient in direction: add explicit `consts_offset >= custom_data_offset` and `model_offset >= consts_offset` ordering checks (addition/range based) and verify `custom_data_offset+custom_data_size <= file_size` and `consts_offset+consts_size <= file_size` (with overflow-safe addition) before using sizes. The fix should be applied identically to BOTH the buffer (157-159) and stream (219-221) copies, which the snippet only shows once. openvinoEp is rejected because the vulnerable parse is OV-internal and the EP does not craft/forward the DataHeader bytes.

## Exploit / Proof of Concept
Craft a 49-byte blob with the following 48-byte DataHeader (little-endian 64-bit fields): custom_data_offset=0x30 (48), custom_data_size=0xFFFFFFFFFFFFFFD0, consts_offset=0x00, consts_size=0x00, model_offset=0x00, model_size=0x00; plus one payload byte. All four `is_valid_model` conditions pass via unsigned wrap-around and `file_size(1) > model_offset(0)`. The subsequent `load_buffer` or `resize` call receives the 18-EiB value, causing DoS or OOB heap access.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for the integer-underflow in
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:158,220
// A crafted 49-byte cache blob with custom_data_offset=48, consts_offset=0,
// custom_data_size=0xFFFFFFFFFFFFFFD0 (= 0 - 48 wrap) passes is_valid_model pre-fix
// and then drives pugixml load_buffer / std::string::resize with ~18 EiB,
// causing bad_alloc/length_error (DoS). After the fix the header must be rejected
// via OPENVINO_ASSERT ("Could not deserialize by device xml header").
//
// SKELETON: ModelDeserializer is an internal intel_cpu class; exact test target,
// include path, and ctor signature must be confirmed against the cpu unit-test tree
// (ov_cpu_unit_tests). Filling the TODOs below is required before this compiles.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// TODO: confirm these headers/paths in src/plugins/intel_cpu/tests/unit/
// #include "utils/graph_serializer/deserializer.hpp"
// #include "openvino/runtime/aligned_buffer.hpp"
// #include "openvino/pass/serialize.hpp"  // ov::pass::StreamSerialize::DataHeader

namespace {

TEST(ModelDeserializerHeaderTest, RejectsUnderflowedCustomDataSize) {
    // 48-byte DataHeader (6 x uint64 little-endian) + 1 payload byte = 49 bytes
    struct RawHeader {
        uint64_t custom_data_offset;
        uint64_t custom_data_size;
        uint64_t consts_offset;
        uint64_t consts_size;
        uint64_t model_offset;
        uint64_t model_size;
    } hdr{};
    hdr.custom_data_offset = sizeof(RawHeader);            // 48 == sizeof(DataHeader)
    hdr.consts_offset      = 0;                             // < custom_data_offset -> underflow
    hdr.custom_data_size   = static_cast<uint64_t>(0) - sizeof(RawHeader); // 0xFFFFFFFFFFFFFFD0
    hdr.consts_size        = 0;                             // == model_offset - consts_offset
    hdr.model_offset       = 0;
    hdr.model_size         = 0;

    std::vector<uint8_t> blob(sizeof(RawHeader) + 1, 0);
    std::memcpy(blob.data(), &hdr, sizeof(RawHeader));

    // TODO: wrap blob in ov::AlignedBuffer and construct ModelDeserializer with a
    //       null/identity CacheDecrypt and a stub ov::ICore, then invoke the
    //       deserialize entry that calls process_model(...).
    // auto buffer = /* make AlignedBuffer over blob */;
    // ov::intel_cpu::ModelDeserializer des(buffer, core, /*decrypt*/{}, false, "");
    // std::shared_ptr<ov::Model> model;
    // Pre-fix: passes is_valid_model, then bad_alloc/length_error from the
    //          18-EiB load_buffer/resize. Post-fix: OPENVINO_ASSERT throws ov::Exception.
    // EXPECT_THROW(des >> model, ov::Exception);
    GTEST_SKIP() << "Fill TODOs: AlignedBuffer wrap + ModelDeserializer ctor/entry.";
}

}  // namespace
```
**Build / run:** Build target: ov_cpu_unit_tests (intel_cpu plugin unit tests, build with -DENABLE_SANITIZER=ON for ASan). Run: ov_cpu_unit_tests --gtest_filter=ModelDeserializerHeaderTest.RejectsUnderflowedCustomDataSize . Expected pre-fix: std::bad_alloc / std::length_error (or ASan allocation-size-too-big) from deserializer.cpp:167/232 because custom_data_size=0xFFFFFFFFFFFFFFD0 bypasses is_valid_model. Expected post-fix: ov::Exception thrown by OPENVINO_ASSERT at deserializer.cpp:160/222.

## Suggested fix
Replace the subtraction-based comparisons with addition-based range checks to avoid unsigned underflow:
```cpp
// Instead of: hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset
// Use:
bool is_valid_model =
    (hdr.custom_data_offset == sizeof(hdr)) &&
    (hdr.consts_offset >= hdr.custom_data_offset) &&
    (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
    (hdr.model_offset >= hdr.consts_offset) &&
    (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
    (file_size > hdr.model_offset) &&
    (hdr.model_offset <= file_size);
```
Additionally, validate `hdr.custom_data_offset + hdr.custom_data_size <= file_size` and `hdr.consts_offset + hdr.consts_size <= file_size` before using these values as buffer access sizes.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #388.
