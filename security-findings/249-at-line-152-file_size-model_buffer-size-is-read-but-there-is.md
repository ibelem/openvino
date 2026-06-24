# Security finding #249: At line 152 `file_size = model_buffer->size()` is read, but there i…

**Summary:** At line 152 `file_size = model_buffer->size()` is read, but there i…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Heap over-read of up to 47 bytes beyond the end of an attacker-supplied AlignedBuffer. Exposes adjacent heap memory contents (potential information disclosure of heap metadata, keys, or other loaded model data), and on some allocator implementations can produce a segfault/crash (DoS). Affected when any caller can pass an undersized cache blob to the CPU EP's import_model path.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:154` — `ModelDeserializer::process_model (AlignedBuffer overload)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled compiled-model cache blob supplied via CoreImpl::load_model_from_cache → plugin.import_model → ModelDeserializer → operator>> → process_model(AlignedBuffer)

## Description / Root cause
At line 152 `file_size = model_buffer->size()` is read, but there is NO check that `file_size >= sizeof(hdr)` (i.e. >= 48 bytes on 64-bit) before the `std::memcpy(&hdr, buffer_base, sizeof hdr)` at line 154. The subsequent `is_valid_model` check at lines 157-160 validates field relationships but is evaluated *after* the memcpy, and also never checks `file_size >= sizeof(DataHeader)`. If the buffer is shorter than 48 bytes the memcpy reads past the end of the heap allocation.

**Validator analysis:** The cited OOB read is real in openvino. At deserializer.cpp:152 file_size is taken but never compared to sizeof(pass::StreamSerialize::DataHeader) before the memcpy at line 154 reads sizeof(hdr) (48 bytes on 64-bit) from buffer_base. The is_valid_model relational checks (custom_data_offset==sizeof(hdr), offset/size arithmetic, file_size>model_offset) all execute at 157-160 AFTER the memcpy and therefore cannot prevent the over-read. CWE-125 is the accurate categorization; impact (heap over-read / info-disclosure / possible crash) is plausible though limited to <=47 bytes adjacent to the AlignedBuffer. The proposed fix (OPENVINO_ASSERT(file_size >= sizeof(hdr), ...) inserted right after line 152) is correct and sufficient: it rejects undersized blobs before any read. Note the sibling istream overload at 206+ uses model_stream.read which is bounded/sets failbit, so it does not over-read the same way — only the AlignedBuffer overload needs the guard. I reject openvinoEp because the vulnerable logic is not duplicated in plugin_impl and EP reachability of this specific mmap overload was not confirmed within budget; the fix belongs in openvino.

## Exploit / Proof of Concept
Supply a cache blob shorter than sizeof(StreamSerialize::DataHeader) == 6*sizeof(size_t) == 48 bytes (e.g. a single-byte buffer). At line 154 the memcpy requests 48 bytes from `buffer_base` but the AlignedBuffer only backs ≤47 bytes; the memcpy reads heap bytes immediately following the allocation. The attacker-read values propagate into `hdr` fields, whose subsequent validation at line 160 will fire an OPENVINO_ASSERT—but only after the over-read has already occurred and potentially leaked heap contents (or crashed if the page boundary is hit).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the missing size guard at
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:152-160
// Pre-fix: ModelDeserializer::process_model(AlignedBuffer) memcpy's sizeof(DataHeader)
//   (48 bytes) from an AlignedBuffer backing fewer bytes -> ASan heap-buffer-overflow.
// Post-fix (OPENVINO_ASSERT(file_size >= sizeof(hdr))): the call cleanly throws ov::Exception
//   before any read, so ASSERT_ANY_THROW passes and ASan stays quiet.
//
// TODO: confirm the intel_cpu unit-test target name (likely ov_cpu_unit_tests) and the
//   include path for ModelDeserializer (deserializer.hpp) from the surrounding tests/ dir.
// TODO: ModelDeserializer's ctor takes a model_buffer, weights cb, decrypt cb, origin weights;
//   verify the exact signature before relying on it.

#include <gtest/gtest.h>
#include <memory>
#include "openvino/runtime/aligned_buffer.hpp"
// TODO: #include the real deserializer header used by intel_cpu graph_serializer

TEST(CpuModelDeserializer, RejectsUndersizedCacheBlobNoOverRead) {
    // A buffer far smaller than sizeof(StreamSerialize::DataHeader) (== 48 bytes on 64-bit).
    constexpr size_t kTiny = 1;
    auto buf = std::make_shared<ov::AlignedBuffer>(kTiny, /*alignment=*/64);
    std::memset(buf->get_ptr(), 0, kTiny);

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with the real ctor and no-op decrypt callbacks,
    //   then invoke the AlignedBuffer process_model overload (directly or via operator>>).
    // Pre-fix this over-reads up to 47 bytes past buf (ASan abort);
    // post-fix it throws on the size guard.
    // ASSERT_ANY_THROW(deserializer.process_model(model, buf));
    GTEST_SKIP() << "TODO: wire up ModelDeserializer ctor + invocation (see header TODOs).";
}
```
**Build / run:** Build: cmake --build build --target ov_cpu_unit_tests ; run: ./bin/ov_cpu_unit_tests --gtest_filter=CpuModelDeserializer.RejectsUndersizedCacheBlobNoOverRead with ASan enabled (-DENABLE_SANITIZER=ON). Pre-fix expectation: AddressSanitizer 'heap-buffer-overflow READ of size 48' originating at deserializer.cpp:154 (std::memcpy into hdr). Post-fix: the call throws ov::Exception ('[CPU] Cache blob too small ...') and the test passes with no ASan report.

## Suggested fix
Insert a size guard immediately after line 152, before the memcpy:
  OPENVINO_ASSERT(file_size >= sizeof(hdr), "[CPU] Cache blob too small to contain StreamSerialize header (" + std::to_string(file_size) + " < " + std::to_string(sizeof(hdr)) + ")");
This ensures the buffer is at least sizeof(DataHeader) bytes before any read occurs, rejecting malformed blobs cleanly rather than triggering an over-read.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #249.
