# Security finding #239: At line 162, `const_type` is read as a raw `uint8_t` (range 0–255) …

**Summary:** At line 162, `const_type` is read as a raw `uint8_t` (range 0–255) …

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** An invalid stored `Type_t` value (26–255) propagates into the weight registry. Any downstream path that calls `get_type_info()` on it will hit `OPENVINO_ASSERT(is_valid_type_idx(type_idx))` at `element_type.cpp:131`; if `OPENVINO_ASSERT` compiles to a no-op in NDEBUG/release builds this becomes an out-of-bounds read on the 26-element static `types_info` array (CWE-125), causing a crash/DoS or information disclosure from adjacent memory. Even with assertions enabled, a malicious local cache file causes an abort in any process that loads it, yielding a reliable DoS against the inference service.
**Affected location:** `targets/openvino/src/inference/src/single_file_storage.cpp:162` — `constant_meta_reader (lambda inside SingleFileStorage::build_content_index)()`
**Validated for repos:** openvino
**Trust boundary:** Locally-cached .blob file on disk → SingleFileStorage::build_content_index → constant_meta_reader lambda

## Description / Root cause
At line 162, `const_type` is read as a raw `uint8_t` (range 0–255) from the untrusted cache file. At line 166 it is immediately cast to `element::Type_t{const_type}` and stored in `m_shared_context->m_weight_registry`. There is zero range validation: no call to `is_valid_type_idx()`, no comparison against `enum_types_size` (= 26). Any byte value 26–255 creates an invalid `Type_t` enumerator that persists in the registry.

**Validator analysis:** The missing-validation flaw is real: at single_file_storage.cpp:162 a single untrusted byte (0-255) is read and at 166 cast straight to element::Type_t{const_type} and stored in the weight registry; the only guard (line 163 `if (s.good())`) checks stream state, not value range, and there is no comparison to enum_types_size / is_valid_type_idx (which lives in an anonymous namespace in element_type.cpp:99-101). CWE-20 is the accurate primary category. However the impact section overstates severity: OPENVINO_ASSERT in OpenVINO (core/except.hpp) is NOT compiled out under NDEBUG — it always evaluates and throws ov::AssertFailure — so get_type_info(Type_t{26}) at element_type.cpp:131 throws rather than performing the OOB `types_info[26]` read claimed as CWE-125. So the realistic impact is a controlled exception → DoS when a downstream path consumes the bad type, gated by a weak local trust boundary (write access to the cache dir). The proposed fix (reject the TLV record when `const_type >= enum_types_size`, exposing enum_types_size/is_valid_type from element_type.hpp) is correct and sufficient; cleanest is to validate immediately after the line-162 read and `return false` before storing into the registry, optionally rejecting the whole record.

## Exploit / Proof of Concept
An attacker with write access to the cache file (any local user, or a compromised model-caching path) crafts a `ConstantMeta` TLV record whose `const_type` byte is set to e.g. 0x1A (26) or higher. `constant_meta_reader` reads it at line 162, skips the only stream-state check (`if (s.good())` at line 163), and stores `element::Type_t{26}` in the registry at line 166 without validation. The next inference call that queries the registry triggers `get_type_info(Type_t{26})`, which in a release (NDEBUG) build executes `types_info[26]` on a 26-element array — one past the end — yielding UB/crash; with assertions enabled it aborts.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/inference/src/single_file_storage.cpp:162-166:
// a ConstantMeta TLV record whose const_type byte is out of range (>= enum_types_size)
// must be rejected by build_content_index instead of being stored as an invalid Type_t.
// Pre-fix: the invalid enum is stored and later get_type_info() (element_type.cpp:131) throws/UB.
// Post-fix: build_content_index returns false / the record is rejected.
//
// TODO: This needs a crafted serialized single-file cache blob (TLV stream) as a fixture.
// TODO: Confirm the exact test target for src/inference (e.g. ov_inference_unit_tests) and the
//       header that exposes SingleFileStorage::build_content_index for direct testing; it may be
//       internal, in which case drive it through the public model-cache load API with a poisoned file.
#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
// #include "single_file_storage.hpp"  // TODO: correct relative include for the internal header

TEST(SingleFileStorageConstantMeta, RejectsOutOfRangeElementType) {
    // TODO: build a minimal TLV stream containing one ConstantMeta record:
    //   source_id(u64), then [const_id(u64), const_offset(u64), const_size(u64), const_type(u8)=0x1A]
    // const_type = 26 (== enum_types_size) is the first invalid Type_t value.
    std::stringstream crafted;
    // TODO: serialize a valid header + ConstantMeta tag/length + payload above into `crafted`.

    // SingleFileStorage storage;
    // const bool ok = storage.build_content_index(crafted);
    // EXPECT_FALSE(ok) << "out-of-range const_type byte must be rejected";
    // ASSERT_TRUE(storage_weight_registry_empty(storage));
    GTEST_SKIP() << "TODO: provide crafted TLV blob fixture and internal accessor";
}
```
**Build / run:** Build target: ov_inference_unit_tests (verify exact name from src/inference/tests). Run: ov_inference_unit_tests --gtest_filter=SingleFileStorageConstantMeta.RejectsOutOfRangeElementType under ASan. Pre-fix expectation: the invalid Type_t{26} is stored and a later get_type_info() call aborts via OPENVINO_ASSERT (element_type.cpp:131) or, with asserts compiled out, ASan reports a global-buffer-overflow read on the 26-entry types_info array. Post-fix: build_content_index returns false (record rejected) and the test passes.

## Suggested fix
Before the `if (s.good())` block at line 163, add a range check: `if (const_type >= static_cast<uint8_t>(element::enum_types_size)) { return false; }`. Expose `enum_types_size` (or an equivalent `element::Type::is_valid_type` predicate) from `element_type.hpp` so the storage layer can validate without depending on internal details. Alternatively call `ov::element::is_valid_type_idx(const_type)` if that helper is made accessible, rejecting the entire TLV record (returning false) on any out-of-range byte.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #239.
