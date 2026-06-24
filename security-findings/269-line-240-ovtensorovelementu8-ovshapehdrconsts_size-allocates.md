# Security finding #269: Line 240: `ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size})…

**Summary:** Line 240: `ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size})…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Any caller loading a malicious cache file (e.g., via ov::Core::compile_model with a crafted blob path) is forced to allocate a heap buffer as large as the attacker's file, enabling controlled OOM/DoS. A 4 GB malicious cache file causes a 4 GB heap allocation on the deserializing machine.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:240` — `ModelDeserializer::process_model (istream overload)()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled hdr.consts_size (size_t) from untrusted cache stream → ov::Tensor heap allocation

## Description / Root cause
Line 240: `ov::Tensor(ov::element::u8, ov::Shape({hdr.consts_size}))` allocates exactly `hdr.consts_size` bytes with no explicit cap smaller than file_size. The only guard is the `is_valid_model` check which only ensures `file_size > hdr.model_offset` (i.e., consts_size < file_size - sizeof(hdr)). There is no limit preventing an attacker from forcing an allocation equal to the size of a multi-gigabyte malicious blob.

**Validator analysis:** The flaw is real. The is_valid_model gate (lines 219-221) enforces only that hdr.consts_size equals the size_t difference model_offset - consts_offset and that file_size > model_offset. It never verifies model_offset >= consts_offset, so the subtraction can underflow: setting custom_data_size=0 (=> consts_offset=sizeof(hdr)) and model_offset=0 yields consts_size = 0 - sizeof(hdr) ≈ 2^64, while file_size need only be > 0. Line 240 then constructs ov::Tensor(u8, Shape({consts_size})) and immediately attempts the allocation before any read, producing a controlled OOM/std::bad_alloc DoS from a tiny crafted blob — a stronger demonstration than the finding's proportional 'allocate == file_size' scenario (which by itself is merely input-proportional and weak). CWE-789 and the DoS impact are accurate. The proposed_fix is directionally correct but insufficient: 'hdr.consts_size + hdr.consts_offset <= file_size' can itself wrap. A correct guard must avoid unsigned underflow/overflow, e.g. OPENVINO_ASSERT(hdr.consts_offset <= file_size && hdr.model_offset >= hdr.consts_offset && hdr.model_offset <= file_size && hdr.consts_size <= file_size - hdr.consts_offset, ...), applied to both this istream overload AND the mmap overload at lines 157-159 which shares the same unchecked arithmetic.

## Exploit / Proof of Concept
Create a cache blob where hdr.consts_size = file_size - sizeof(DataHeader) - 1 (e.g., 2 GB). All is_valid_model conditions are satisfied. Line 240 allocates a 2 GB ov::Tensor. No stream content is actually validated before the allocation occurs.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 in
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:240 (is_valid_model gate at 219-221)
// Pre-fix: a crafted StreamSerialize blob whose header has model_offset < consts_offset
// makes hdr.consts_size underflow (size_t) to ~2^64; line 240 attempts a multi-GB/2^64
// ov::Tensor allocation -> std::bad_alloc / OOM before any stream read.
// Post-fix: the bounds/underflow check rejects the header via ov::Exception.
//
// NOTE: ModelDeserializer + pass::StreamSerialize::DataHeader are intel_cpu-internal; this is a
// SKELETON because the exact DataHeader field layout and the deserializer entry symbol must be
// confirmed from the source before this will compile.
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
// TODO: include the real headers, e.g.
//   #include "utils/serialize.hpp"            // pass::StreamSerialize::DataHeader
//   #include "utils/graph_serializer/deserializer.hpp"  // ov::intel_cpu::ModelDeserializer

TEST(IntelCpuModelDeserializer, RejectsUnderflowedConstsSize) {
    // TODO: replace with the real DataHeader type/layout from serialize.hpp
    struct DataHeaderMock {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};

    hdr.custom_data_offset = sizeof(DataHeaderMock); // must == sizeof(hdr)
    hdr.custom_data_size   = 0;                      // => consts_offset == custom_data_offset
    hdr.consts_offset      = sizeof(DataHeaderMock);
    hdr.model_offset       = 0;                      // model_offset < consts_offset -> underflow
    hdr.consts_size        = hdr.model_offset - hdr.consts_offset; // wraps to ~2^64
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.resize(sizeof(hdr) + 16, '\0'); // file_size > model_offset, tiny file
    std::stringstream ss(blob);

    std::shared_ptr<ov::Model> model;
    // TODO: construct the deserializer exactly as production code does and invoke the istream overload:
    //   ov::intel_cpu::ModelDeserializer d(ss, /*decrypt*/{}, /*model_loader*/{}, /*orig_weights*/nullptr);
    //   EXPECT_THROW(d >> model, ov::Exception);   // post-fix: header rejected, no huge allocation
    // Pre-fix this path reaches make_shared<ov::Tensor>(u8,{~2^64}) -> std::bad_alloc (ASan/OOM).
    GTEST_SKIP() << "TODO: wire up real ModelDeserializer entry point and DataHeader layout";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (intel_cpu unit tests). Run: ov_cpu_unit_tests --gtest_filter=IntelCpuModelDeserializer.RejectsUnderflowedConstsSize . Pre-fix expectation: crafted header reaches deserializer.cpp:240 and triggers std::bad_alloc / ASan allocation-size-too-big (or OOM kill) due to underflowed hdr.consts_size; post-fix expectation: header is rejected with ov::Exception from the added bounds/underflow guard before any allocation.

## Suggested fix
Before line 240, add an explicit size cap: `OPENVINO_ASSERT(hdr.consts_size <= some_reasonable_limit && hdr.consts_size + hdr.consts_offset <= file_size - hdr_pos, "[CPU] consts_size exceeds stream bounds");`. Additionally, consider a compile-time or runtime configurable maximum model weight size (e.g., 2 GB) to prevent deliberate OOM via oversized cache files.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #269.
