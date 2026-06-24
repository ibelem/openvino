# Security finding #252: Line 212 captures `file_size` as the **absolute** stream-end positi…

**Summary:** Line 212 captures `file_size` as the **absolute** stream-end positi…

**CWE IDs:** CWE-20: Improper Input Validation (absolute-vs-relative coordinate mismatch)
**Severity / Impact:** When `hdr_pos > 0` (stream is mid-buffer, e.g., a cached blob prepended with metadata), an attacker who controls the blob header can: (1) cause `xml_string->resize(hdr.model_size)` at line 249 to allocate a large buffer, then `model_stream.read(...)` at line 250 silently reads 0 bytes (stream seeked past EOF) leaving the buffer filled with uninitialized/stale heap bytes; (2) cause a similarly oversized `ov::Tensor` allocation at line 240 (`hdr.consts_size` is also validated against the inflated `file_size`), with line 243's read returning partial/zero bytes. The uninitialized XML string is then fed to pugixml (line 250→`create_ov_model`), which may crash or be exploited. The uninitialized weight tensor silently corrupts inference. In both cases `std::bad_alloc` is possible if the inflated size is large.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:221` — `ModelDeserializer::process_model(std::shared_ptr<ov::Model>&, std::reference_wrapper<std::istream>)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted cache/blob istream passed to process_model → header field validation → allocation and memcpy

## Description / Root cause
Line 212 captures `file_size` as the **absolute** stream-end position (`model_stream.tellg()` after seekg to end), while all header offsets (`hdr.model_offset`, `hdr.consts_offset`, etc.) are **relative** to `hdr_pos` (line 210). The guard at line 221 (`file_size > hdr.model_offset`) therefore compares an absolute end against a relative offset: when `hdr_pos > 0`, it is satisfied by any `hdr.model_offset < file_size`, even if `hdr.model_offset + hdr_pos > file_size` (i.e., the data region lies entirely past end-of-stream). Downstream at line 224, `hdr.model_size = file_size - hdr.model_offset` is similarly computed with mismatched bases, producing an inflated size.

**Validator analysis:** The core code defect is REAL and confirmed by reading deserializer.cpp:206-275: line 210 captures hdr_pos, line 212 captures file_size as the ABSOLUTE end position, yet all subsequent seeks (227 custom_data_offset+hdr_pos, 241 consts_offset+hdr_pos, 248 model_offset+hdr_pos) treat the header offsets as RELATIVE to hdr_pos. The validity guard (line 221: file_size > hdr.model_offset) and the size computation (line 224: hdr.model_size = file_size - hdr.model_offset) therefore mix an absolute base with relative offsets. When hdr_pos>0 (explicitly anticipated — see the mmap-variant comment at line 149 'Blob from cache may have other header'), the read at line 250 seeks to model_offset+hdr_pos and asks for model_size = file_size-model_offset bytes, which is hdr_pos bytes more than actually remain, so the tail of xml_string is left unfilled. CWE-20 (improper input validation, absolute-vs-relative mismatch) is the correct category. HOWEVER the stated impact is partly overstated: (a) std::string::resize value-initializes to '\0', so the unfilled tail is ZERO bytes, not 'uninitialized/stale heap' — there is no heap info-leak; (b) consts_size is validated against RELATIVE offsets (hdr.model_offset - hdr.consts_offset at line 221), not against the inflated file_size, so the data_blob/Tensor allocation is NOT affected by this bug; (c) model_size <= file_size (the real stream length), so the over-allocation is bounded by actual stream size, not arbitrarily large — bad_alloc is unlikely beyond the genuine blob size. Real consequence: malformed/zero-padded XML fed to pugixml (which then fails its load and trips OPENVINO_ASSERT) plus a short read with no model_stream.good() check — i.e. a robustness/DoS-grade input-validation flaw, not memory corruption. The proposed fix is CORRECT and sufficient for the size arithmetic: compute avail_size = file_size - hdr_pos, use avail_size in the line-221 guard and the line-224 size, add hdr_pos overflow guards before the seek adds, and check model_stream.good() after each read/seekg. Note the mmap/buffer variant (line 145) is already consistent (uses model_buffer->size() with no hdr_pos) and is NOT vulnerable.

## Exploit / Proof of Concept
Craft a blob with `hdr_pos = P > 0` (achieved by wrapping it in a larger stream or using a seekable cache stream already advanced to offset P). Set `hdr.custom_data_offset = sizeof(DataHeader) = 48`, `hdr.custom_data_size = 0`, `hdr.consts_offset = 48`, `hdr.consts_size = S`, `hdr.model_offset = 48 + S`. Choose S such that `48 + S < file_size` (passes the absolute check at line 221) but `48 + S + P > file_size` (actual model data is past end). `hdr.model_size` is then computed as `file_size - (48+S)`, a large value. `xml_string->resize(hdr.model_size)` allocates that many bytes; `seekg((48+S)+P)` lands past EOF; `model_stream.read(...)` fails silently; `xml_string` contains garbage which pugixml parses, crashing or consuming attacker-influenced data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for absolute-vs-relative base mismatch at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:212,221,224.
// Pre-fix: when the istream is positioned at hdr_pos>0, file_size (absolute) is mixed
// with relative header offsets, so hdr.model_size is inflated by hdr_pos and the final
// read (line 250) runs past EOF leaving a zero-padded buffer / failing pugixml parse.
// Post-fix: the deserializer must reject the blob (OPENVINO_ASSERT) or size model_size
// using avail_size = file_size - hdr_pos so no over-read occurs.
//
// TODO: confirm the exact target/binary (expected: ov_cpu_unit_tests) and that
//       ModelDeserializer / pass::StreamSerialize::DataHeader are reachable from the
//       unit-test include path. ModelDeserializer is an internal CPU-plugin symbol;
//       if it is not exported to tests, exercise the bug through ov::Core::import_model
//       on a CPU-plugin blob whose stream is pre-advanced by P>0 bytes instead.
//
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "openvino/core/except.hpp"
// TODO: include the real headers for ModelDeserializer and StreamSerialize::DataHeader
//       (e.g. utils/graph_serializer/deserializer.hpp and the serialize pass header).

TEST(intel_cpu_ModelDeserializer, RejectsHeaderOffsetsPastEofWhenStreamPreAdvanced) {
    using DataHeader = ov::pass::StreamSerialize::DataHeader;  // TODO: verify namespace

    // Build a blob: [P prefix bytes][DataHeader][consts][model], but truncate so that
    // model_offset+hdr_pos lies past EOF while file_size > model_offset still holds.
    const size_t P = 64;                 // hdr_pos prefix (simulates cache metadata)
    const size_t S = 16;                 // consts_size
    DataHeader hdr{};
    hdr.custom_data_offset = sizeof(DataHeader);
    hdr.custom_data_size   = 0;
    hdr.consts_offset      = sizeof(DataHeader);
    hdr.consts_size        = S;
    hdr.model_offset       = sizeof(DataHeader) + S;
    // hdr.model_size is recomputed by the deserializer.

    std::string buf;
    buf.append(P, '\xAB');                                   // prefix => hdr_pos == P
    buf.append(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    buf.append(S, '\x00');                                  // consts region
    // Intentionally provide FEWER model bytes than model_size-relative would require,
    // so that model_offset + hdr_pos is past the real end of the (relative) data.
    buf.append(4, '<');                                     // tiny, truncated 'model'

    std::istringstream ss(buf, std::ios::binary);
    ss.seekg(static_cast<std::streamoff>(P));               // advance to hdr_pos = P

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with no-op cache_decrypt / create_ov_model hook.
    // ModelDeserializer des(...);
    // Pre-fix this either over-reads (short read, zero-padded XML) and asserts inside
    // pugixml, or silently builds a corrupt model. Post-fix it must reject cleanly.
    // EXPECT_THROW(des.process_model(model, std::ref(ss)), ov::Exception);
    GTEST_SKIP() << "TODO: wire ModelDeserializer ctor + create_ov_model callback; "
                    "assert EXPECT_THROW(process_model(model, std::ref(ss)), ov::Exception).";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run: ov_cpu_unit_tests --gtest_filter='intel_cpu_ModelDeserializer.RejectsHeaderOffsetsPastEofWhenStreamPreAdvanced'. Build with -DENABLE_SANITIZER=ON. Pre-fix expectation: a short read past EOF leaves model_size-hdr_pos trailing zero bytes and pugixml load fails / OPENVINO_ASSERT fires unpredictably (or, with mismatched arithmetic, an out-of-range seek). Post-fix expectation: process_model rejects the blob with ov::Exception (EXPECT_THROW passes) because model_size is computed from avail_size = file_size - hdr_pos. Replace GTEST_SKIP with the EXPECT_THROW once ModelDeserializer is wired.

## Suggested fix
Normalize all arithmetic to use the relative available size, not the absolute `file_size`. Replace line 212–221 with: `const size_t avail_size = file_size - hdr_pos;` and then check `avail_size > hdr.model_offset` at line 221. Replace line 224 with `hdr.model_size = avail_size - hdr.model_offset;`. Also add overflow guard before the add at seek sites (lines 227, 241, 248): assert `hdr.model_offset <= SIZE_MAX - hdr_pos`. Additionally, check `model_stream.good()` after each `read()` and `seekg()` call before using the data.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #252.
