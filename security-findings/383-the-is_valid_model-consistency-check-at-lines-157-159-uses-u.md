# Security finding #383: The `is_valid_model` consistency check at lines 157-159 uses unsign…

**Summary:** The `is_valid_model` consistency check at lines 157-159 uses unsign…

**CWE IDs:** CWE-191: Integer Underflow / CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds heap read: pugixml attempts to iterate `SIZE_MAX - 47` bytes starting at `buffer_base + 48`. On any real system this reads far past the allocated region, causing an immediate SIGSEGV / access violation (process crash, DoS). On systems with adjacent mappings the OOB read could also leak heap or stack contents before the crash.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:157` — `ModelDeserializer::process_model(std::shared_ptr<ov::AlignedBuffer>)()`
**Validated for repos:** openvino
**Trust boundary:** attacker-writable filesystem cache blob → import_model → ModelDeserializer::process_model(AlignedBuffer)

## Description / Root cause
The `is_valid_model` consistency check at lines 157-159 uses unsigned subtraction (`hdr.consts_offset - hdr.custom_data_offset` and `hdr.model_offset - hdr.consts_offset`) without first verifying that the subtrahend is ≤ the minuend. All six `DataHeader` fields are attacker-controlled `size_t` values read verbatim from the cache blob at line 154. An attacker who sets `consts_offset = 0` and `custom_data_offset = sizeof(hdr) = 48` causes the unsigned subtraction `0 - 48` to wrap to `SIZE_MAX - 47`. If `custom_data_size` is also set to `SIZE_MAX - 47`, check #2 passes. Combined with a small `model_offset` (> 0) satisfying check #4, `is_valid_model` evaluates to `true` despite the wrapped sizes. Line 167 then calls `xml_in_out_doc.load_buffer(buffer_base + 48, SIZE_MAX - 47, ...)`, instructing pugixml to read ~18.4 exabytes past the actual buffer end.

**Validator analysis:** The integer-underflow defect (CWE-191) is real and confirmed at deserializer.cpp:157-159: the consistency check subtracts attacker-controlled size_t header fields without first checking subtrahend<=minuend, so wrap-around lets is_valid_model pass with inconsistent offsets/sizes. The trust boundary (attacker-writable cache blob read via memcpy at :154 into a struct of verbatim fields) is genuine. The proposed fix (explicit >= ordering guards before the equality/size checks) is correct and sufficient, eliminating the wrap. One caveat on the stated IMPACT/exploit: the specific demonstration claims load_buffer(buffer_base+48, SIZE_MAX-47) at :167 performs an 18EB OOB read — but pugixml's load_buffer (the copying, non-inplace variant) allocates `size` bytes first; a ~SIZE_MAX allocation fails and returns status_out_of_memory, tripping the OPENVINO_ASSERT at :171 (handled exception), so that exact path is a DoS/abort rather than a true OOB read. The more reliable OOB read from the same underflow is the consts path: setting consts_offset>model_offset wraps hdr.consts_size to a huge value, and the SharedBuffer weights_buf created at :177-180 (buffer_base+consts_offset, huge_size) is later dereferenced during create_ov_model weight reads — a genuine out-of-bounds heap read. So CWE-191 is accurate, CWE-125 is achievable (via the consts/weights path more than the cited custom_data path), and the impact 'OOB read / crash' holds. Fix stands.

## Exploit / Proof of Concept
Craft a cache blob where the 48-byte `DataHeader` has: `custom_data_offset = 48`, `custom_data_size = SIZE_MAX - 47`, `consts_offset = 0`, `consts_size = 8`, `model_offset = 8`, `model_size` (ignored, recomputed). The total file need only be 9+ bytes. Feed this as the cache blob to `import_model`; the path reaches `process_model(AlignedBuffer)` at line 154, the `is_valid_model` check at 157-160 passes (unsigned wrap makes both sides equal), and `load_buffer` at line 167 is called with size `SIZE_MAX - 47`, triggering the OOB read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for the integer-underflow validation bypass at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:157-159.
// Pre-fix: a crafted DataHeader with consts_offset < custom_data_offset makes
//   (consts_offset - custom_data_offset) underflow, custom_data_size set to the
//   wrapped value satisfies check #2, is_valid_model becomes true, and the path
//   proceeds to construct over-sized buffers / OOB read.
// Post-fix (>= ordering guards): is_valid_model is false and process_model throws
//   ov::Exception via OPENVINO_ASSERT at :160 before any unsafe buffer use.
//
// Harness: ov_cpu_unit_tests (the intel_cpu component's gtest target).
// NOTE: ModelDeserializer / pass::StreamSerialize::DataHeader and the AlignedBuffer
//   wrapping constructor symbols must be confirmed against the real headers — TODOs below.

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

// TODO: include the real headers, e.g.:
//   #include "utils/graph_serializer/deserializer.hpp"
//   #include "openvino/pass/serialize.hpp"   // for pass::StreamSerialize::DataHeader
//   #include "openvino/runtime/aligned_buffer.hpp"

TEST(CpuModelDeserializer, RejectsUnderflowingDataHeader) {
    // sizeof(DataHeader) is 6 * sizeof(size_t) == 48 on LP64.
    // TODO: replace with the real pass::StreamSerialize::DataHeader type & field order.
    struct DataHeaderLike {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};
    static_assert(sizeof(DataHeaderLike) == 48, "layout assumption");

    hdr.custom_data_offset = sizeof(DataHeaderLike);          // == 48, passes check #1
    hdr.consts_offset      = 0;                               // < custom_data_offset -> underflow
    hdr.custom_data_size   = hdr.consts_offset - hdr.custom_data_offset; // wraps to SIZE_MAX-47
    hdr.model_offset       = 8;
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset;       // == 8
    hdr.model_size         = 0;

    std::vector<char> blob(64, 0);
    std::memcpy(blob.data(), &hdr, sizeof(hdr));

    // TODO: wrap `blob` in an ov::AlignedBuffer (or the project's SharedBuffer over a
    //       std::vector) so model_buffer->get_ptr()/size() return blob.data()/blob.size().
    // std::shared_ptr<ov::AlignedBuffer> buf = make_aligned_buffer(blob.data(), blob.size());
    //
    // ModelDeserializer deserializer(/* ctor args: model_buffer, extension, decrypt cbs */);
    // std::shared_ptr<ov::Model> model;
    // EXPECT_THROW(deserializer.process_model(model, buf), ov::Exception);
    GTEST_SKIP() << "TODO: finalize ModelDeserializer ctor/process_model invocation and AlignedBuffer wrapper";
}
```
**Build / run:** Build: cmake --build . --target ov_cpu_unit_tests. Run: ./ov_cpu_unit_tests --gtest_filter=CpuModelDeserializer.RejectsUnderflowingDataHeader. Pre-fix expectation: is_valid_model wrongly true → no throw (and under ASan, downstream over-sized buffer reads report heap-buffer-overflow); post-fix: OPENVINO_ASSERT at deserializer.cpp:160 throws ov::Exception so EXPECT_THROW passes.

## Suggested fix
Replace the current unsigned-subtraction consistency check with explicit ordering checks before computing sizes. For example:
```cpp
bool is_valid_model =
    (hdr.custom_data_offset == sizeof(hdr)) &&
    (hdr.consts_offset  >= hdr.custom_data_offset) &&
    (hdr.model_offset   >= hdr.consts_offset)  &&
    (file_size          >= hdr.model_offset)   &&
    (hdr.custom_data_size == hdr.consts_offset  - hdr.custom_data_offset) &&
    (hdr.consts_size      == hdr.model_offset   - hdr.consts_offset)      &&
    (hdr.model_offset + (file_size - hdr.model_offset) == file_size);
```
The ordering guards (`>=`) prevent the unsigned underflow before the equality checks, eliminating the wraparound attack surface.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #383.
