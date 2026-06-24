# Security finding #253: After each `model_stream.seekg(...)` and `model_stream.read(...)` c…

**Summary:** After each `model_stream.seekg(...)` and `model_stream.read(...)` c…

**CWE IDs:** CWE-908: Use of Uninitialized Resource / CWE-754: Improper Check for Unusual Conditions
**Severity / Impact:** Even absent the absolute/relative mismatch in Finding 1, a corrupted or truncated blob can cause short reads that leave the weight tensor and XML string buffers holding stale heap memory. The XML parser (`create_ov_model`) then processes attacker-influenced/uninitialized bytes, potentially crashing (DoS) or leaking heap data back through error messages. If the partial-read tensor is used as model weights, inference output is silently corrupted.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:240` — `ModelDeserializer::process_model(std::shared_ptr<ov::Model>&, std::reference_wrapper<std::istream>)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled istream blob → stream read operations with no I/O error checking

## Description / Root cause
After each `model_stream.seekg(...)` and `model_stream.read(...)` call (lines 233, 241–243, 248–250), the stream state is never checked. `std::istream::read` sets `failbit`/`eofbit` on short reads (e.g., stream seeked past end) but does not throw by default. Line 240 allocates an `ov::Tensor` of `hdr.consts_size` bytes, line 249 resizes `xml_string` to `hdr.model_size` bytes; if either `seekg` silently fails (lands past EOF), the subsequent `read` writes 0 bytes into the allocated buffer, leaving it with uninitialized heap content. The `xml_string` buffer is then handed to `create_ov_model` (line 270) and the tensor to model weights.

**Validator analysis:** The cited code at deserializer.cpp:216-250 reads a header then performs seekg/read with no stream-state validation. CWE-908 is ACCURATE for the constants tensor: data_blob (line 240) is an ov::Tensor whose backing buffer is not value-initialized, so a short read at line 243 leaves stale heap bytes that become model weights. The finding's XML-buffer claim is WRONG: xml_string->resize(hdr.model_size) at line 249 value-initializes the std::string to '\0', so a short read there leaves zeros, not uninitialized memory — no info leak from that path. Crucially the finding's assertion that this is exploitable 'even absent the absolute/relative mismatch (Finding 1)' is FALSE: with hdr_pos==0 the is_valid_model check (220-221) ties consts_offset+consts_size==model_offset<file_size and model_size==file_size-model_offset, so every read is in-bounds and no short read can occur. The short read is reachable ONLY when hdr_pos>0 (the relative-offset header vs absolute file_size mismatch), so this finding is largely a derivative framing of Finding 1, but the missing I/O check IS a genuine defect on that reachable path. The proposed fix (OPENVINO_ASSERT(model_stream.good() && gcount()==expected) after each read) is correct and sufficient as defense-in-depth; it should be paired with the deeper fix of validating offsets against file_size including hdr_pos (i.e., checking model_offset+hdr_pos+model_size<=file_size).

## Exploit / Proof of Concept
Supply a well-formed header that passes `is_valid_model` but whose `consts_offset + hdr_pos` points near the end of the file, so that `hdr.consts_size` bytes cannot all be read. The `ov::Tensor` at line 240 is allocated for the full `hdr.consts_size` bytes; the read at line 243 returns fewer bytes (stream hits EOF); `model_stream.fail()` is set but never checked; the remainder of the tensor buffer holds uninitialised heap content that becomes the model's constant weights.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:240-243
// Pre-fix: a cache blob whose header offsets, combined with a nonzero stream start position (hdr_pos),
// place consts_offset+hdr_pos+consts_size beyond EOF causes std::istream::read to short-read into the
// non-zero-initialized ov::Tensor weight buffer with NO stream-state check (CWE-908). ASan/MSan will flag
// the uninitialized read in create_ov_model, or the assertion below will fail.
// Post-fix: process_model must check model_stream.good()/gcount() and throw ov::Exception on a short read.
//
// SKELETON — exact symbols/headers must be confirmed against intel_cpu's ov_cpu_unit_tests tree before use.
#include <gtest/gtest.h>
#include <sstream>
#include <string>
// TODO: include the real deserializer header, e.g.
// #include "utils/graph_serializer/deserializer.hpp"
// #include "openvino/runtime/core.hpp"

TEST(IntelCpuModelDeserializer, ShortReadOnTruncatedConstsBlobThrows) {
    // TODO: build a byte buffer that:
    //  (1) begins with a prefix so that hdr_pos > 0 when the deserializer reads the header,
    //  (2) carries a pass::StreamSerialize::DataHeader whose custom_data_offset==sizeof(hdr),
    //      custom_data_size==consts_offset-custom_data_offset, consts_size==model_offset-consts_offset,
    //      and file_size>model_offset (so is_valid_model passes at deserializer.cpp:219-221),
    //  (3) is physically truncated so that consts_offset+hdr_pos+consts_size > actual file size,
    //      forcing the read at line 243 to short-read.
    // TODO: replace with the real header struct layout and a helper that emits the crafted blob.
    std::string crafted_blob = /* TODO: crafted_cache_blob_with_short_consts() */ std::string();
    std::istringstream stream(crafted_blob, std::ios::binary);
    // stream.seekg(prefix_len); // make hdr_pos > 0

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer the way ov_cpu_unit_tests does and invoke process_model(model, stream).
    // EXPECT_THROW(deserializer.process_model(model, std::ref(stream)), ov::Exception);
    GTEST_SKIP() << "TODO: supply crafted DataHeader blob + ModelDeserializer construction for intel_cpu";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run: ov_cpu_unit_tests --gtest_filter=IntelCpuModelDeserializer.ShortReadOnTruncatedConstsBlobThrows . Pre-fix expectation: ASan/MSan 'use-of-uninitialized-value' (or heap read) inside create_ov_model from the partially-read ov::Tensor at deserializer.cpp:243, or no exception thrown. Post-fix expectation: process_model throws ov::Exception ("[CPU] Short read on consts data") and the test passes.

## Suggested fix
After every `model_stream.read(...)` call, verify `model_stream.good()` (or check `model_stream.gcount() == expected`) and throw / assert on failure: e.g., `model_stream.read(..., hdr.consts_size); OPENVINO_ASSERT(model_stream.good() && model_stream.gcount() == static_cast<std::streamsize>(hdr.consts_size), "[CPU] Short read on consts data");`. Apply the same guard after the custom-data read (line 233) and the XML read (line 250).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #253.
