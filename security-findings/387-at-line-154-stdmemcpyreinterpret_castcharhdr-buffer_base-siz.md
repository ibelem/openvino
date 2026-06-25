# Security finding #387: At line 154, `std::memcpy(reinterpret_cast<char*>(&hdr), buffer_bas…

**Summary:** At line 154, `std::memcpy(reinterpret_cast<char*>(&hdr), buffer_bas…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** If an attacker supplies a blob smaller than 48 bytes (e.g., a 1-byte cache file), the memcpy reads 47 bytes past the end of the allocated buffer, causing a heap out-of-bounds read. This can leak adjacent heap contents (information disclosure) or trigger a crash (DoS). Any caller path that loads from a cache — including `Plugin::import_model` via `plugin.cpp:762` — reaches this code.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:154` — `ModelDeserializer::process_model(std::shared_ptr<ov::AlignedBuffer>)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled model cache blob (AlignedBuffer) passed to import_model / ModelDeserializer

## Description / Root cause
At line 154, `std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base, sizeof hdr)` reads `sizeof(DataHeader)` = 48 bytes (on 64-bit LP64) from `buffer_base` without first verifying that `model_buffer->size() >= sizeof(hdr)`. The `file_size` variable is computed at line 152 but is never compared against `sizeof(hdr)` before the memcpy.

**Validator analysis:** The CWE-125 OOB-read claim is accurate for the AlignedBuffer overload: lines 150-160 show file_size is computed (152) but the only validation occurs AFTER the memcpy (157-160, is_valid_model), so an attacker-supplied cache blob smaller than sizeof(DataHeader) causes the memcpy at 154 to read past the allocation — a heap OOB read (info-leak/DoS). Impact (info disclosure / crash) is accurate. The proposed fix (OPENVINO_ASSERT(file_size >= sizeof(hdr)) before the memcpy) is correct and sufficient for the mmap path. Note the second part of the proposed fix for the stream path (line 215) is largely unnecessary: std::istream::read into &hdr on an undersized stream reads only the available bytes and sets failbit rather than reading OOB, so that path is not a memory-safety bug (though adding a gcount()/failbit check there is still good hygiene to reject truncated blobs deterministically). Reachability from the EP boundary is unconfirmed because the EP's import_model path uses the istream overload (safe-ish stream read), not the AlignedBuffer memcpy — hence openvinoEp is rejected while the underlying openvino defect is validated.

## Exploit / Proof of Concept
Supply a model cache blob (via shared NFS/container cache, a malicious .blob file, or an overwritten OVMS cache entry) that is fewer than 48 bytes. `model_buffer->size()` returns e.g. 1; `std::memcpy` reads 47 bytes past the buffer's allocation, causing OOB heap read. The `is_valid_model` check at line 157 is reached with garbage `hdr` fields and will likely assert-fail, but the OOB read has already occurred.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-125 OOB read at
// openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:154
// Pre-fix: ModelDeserializer::process_model(AlignedBuffer) memcpy's sizeof(DataHeader)
// (~48 bytes) out of a buffer whose size() may be < sizeof(hdr) -> heap OOB read
// (ASan: heap-buffer-overflow READ). Post-fix: an explicit size guard throws ov::Exception
// ("Cache blob too small...") before the memcpy, so EXPECT_THROW passes cleanly.
//
// SKELETON: ModelDeserializer is an internal CPU-plugin type; exact include path,
// constructor signature, and the AlignedBuffer concrete type must be confirmed against
// the source tree before this compiles.

#include <gtest/gtest.h>
#include <memory>
#include "openvino/runtime/aligned_buffer.hpp"   // TODO: confirm header for ov::AlignedBuffer
// TODO: include the real header that declares intel_cpu ModelDeserializer
// e.g. "utils/graph_serializer/deserializer.hpp" (path relative to intel_cpu/src)

using namespace ov;

TEST(cpu_model_deserializer, undersized_cache_blob_is_rejected_not_oob) {
    // 1-byte buffer, far smaller than sizeof(pass::StreamSerialize::DataHeader) (~48 bytes)
    auto tiny = std::make_shared<ov::AlignedBuffer>(/*size=*/1);  // TODO: confirm ctor
    reinterpret_cast<char*>(tiny->get_ptr())[0] = 0x00;

    std::shared_ptr<ov::Model> model;
    // TODO: confirm ModelDeserializer ctor args (decrypt cfg / origin weights may be required)
    // intel_cpu::ModelDeserializer des(/*...*/);
    // Pre-fix this memcpy reads 47 bytes OOB (ASan trap); post-fix it must throw.
    // EXPECT_THROW(des.process_model(model, tiny), ov::Exception);
    GTEST_SKIP() << "TODO: wire ModelDeserializer ctor + process_model(AlignedBuffer) overload";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (intel_cpu plugin unit harness). Run: ov_cpu_unit_tests --gtest_filter=cpu_model_deserializer.undersized_cache_blob_is_rejected_not_oob. Build with -DENABLE_SANITIZER=ON (ASan). Pre-fix expectation: ASan 'heap-buffer-overflow READ of size 48' inside ModelDeserializer::process_model at deserializer.cpp:154. Post-fix expectation: ov::Exception ('[CPU] Cache blob too small to contain DataHeader') and the test passes.

## Suggested fix
Add an explicit size guard immediately after computing `file_size`:
```cpp
OPENVINO_ASSERT(file_size >= sizeof(hdr),
    "[CPU] Cache blob too small to contain DataHeader (", file_size, " < ", sizeof(hdr), ")");
```
Insert this check between lines 152 and 153 (before the `memcpy`). Apply the same guard to the stream-based path at deserializer.cpp:215 by checking `(file_size - hdr_pos) >= sizeof(hdr)` before the `model_stream.read`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #387.
