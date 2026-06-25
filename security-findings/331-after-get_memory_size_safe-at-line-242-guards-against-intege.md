# Security finding #331: After `get_memory_size_safe` at line 242 guards against integer ove…

**Summary:** After `get_memory_size_safe` at line 242 guards against integer ove…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Any consumer that deserializes an XML/IR model (e.g. `ov::Core::read_model`) with an attacker-controlled Constant node shape can be forced to attempt a multi-gigabyte heap allocation, exhausting system memory (DoS / OOM kill). On systems where the allocator returns null instead of throwing (see CWE-476 finding below), this leads to a null-pointer dereference crash immediately after.
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:242` — `Constant::allocate_buffer()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied XML model attributes `element_type` and `shape` ingested via `visit_attributes` (lines 544-545) with no prior validation

## Description / Root cause
After `get_memory_size_safe` at line 242 guards against integer overflow and the `OPENVINO_ASSERT` at line 243 guards against a null optional, `*byte_size` is passed completely uncapped to `std::make_shared<AlignedBuffer>(*byte_size, host_alignment())` at line 249. A non-overflowing but enormous value (e.g. shape=[2^31], element_type=f32 → 8 GB; shape=[2^33], element_type=u8 → 8 GB) sails through both checks and reaches the allocator with no upper bound enforced anywhere on the call path.

**Validator analysis:** The cited data flow is confirmed: visit_attributes deserializes attacker-controlled element_type/shape (constant.cpp:544-545) and calls allocate_buffer (551) which computes byte_size from shape alone and allocates an AlignedBuffer of that size with no upper bound and no comparison against the actual value-buffer/file size (249). get_memory_size_safe (242) only rejects multiplication overflow (returns nullopt) — a non-overflowing 8 GB request passes. Because the buffer is sized purely from the declared shape *before* the value blob is read, a tiny IR can demand a multi-gigabyte allocation, so CWE-789 (allocation with excessive size) is accurately categorised and the path is reachable via ov::Core::read_model on an untrusted IR. The 'impact' claim is partially overstated: aligned_alloc failure throws std::bad_alloc, which propagates as an exception out of read_model (a controlled failure / OOM-DoS), so the chained CWE-476 null-deref is speculative and depends on a separate, unproven finding. The proposed fix is INCORRECT/insufficient: a hard `MAX_CONSTANT_BYTES = 1 GB` cap would reject legitimate large constants (e.g. LLM weight tensors routinely exceed 1 GB) and break normal model loading. The correct fix is to validate the declared byte_size against the actual available data — i.e. cross-check shape-derived size against the size of the backing value buffer / mmapped weights region during deserialization, rejecting only when the declaration exceeds what the input actually provides, rather than imposing an arbitrary absolute ceiling.

## Exploit / Proof of Concept
Craft an IR XML file with a `Const` node containing `element_type="f32"` and `shape="2147483648"` (2^31 elements). During deserialization: `visit_attributes` stores these into `m_element_type`/`m_shape` at lines 544-545 → `allocate_buffer` is called at line 551 → `get_memory_size_safe` returns `optional(8589934592)` (8 GB, no overflow on 64-bit) → OPENVINO_ASSERT passes → `AlignedBuffer(8589934592, ...)` is called → `util::aligned_alloc(8 GB, ...)` is attempted. No rejection at any point.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/core/src/op/constant.cpp:242-249
// (Constant::allocate_buffer): a Constant whose declared shape implies a
// byte size far larger than any backing data must be rejected with an
// ov::Exception instead of being passed uncapped to AlignedBuffer.
//
// Pre-fix: constructing the Constant attempts a multi-GB aligned_alloc
//          (succeeds and wastes memory, or throws std::bad_alloc, NOT ov::Exception).
// Post-fix: deserialization/allocation validates the declared size against the
//           available data and throws ov::Exception.
//
// NOTE: This is a SKELETON. The real trigger is deserialization of an IR where
// the declared shape exceeds the actual <value>/weights blob; constructing a
// Constant directly from a Shape does not model that mismatch and may simply
// allocate. The TODOs below name what is missing.

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"

using namespace ov;

TEST(constant, reject_excessive_declared_size_vs_data) {
    // TODO: replace direct construction with a read_model() of a crafted IR
    //       (.xml) whose <Const> declares element_type="f32" shape="2147483648"
    //       but supplies a tiny/empty backing buffer, so the fix can compare
    //       declared byte_size against the actual data size.
    const Shape excessive_shape{static_cast<size_t>(1) << 31};  // 2^31 f32 = 8 GB
    EXPECT_THROW(op::v0::Constant(element::f32, excessive_shape), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests. Run: ov_core_unit_tests --gtest_filter=constant.reject_excessive_declared_size_vs_data . Pre-fix this either allocates ~8 GB (no throw / OOM under ASan) or throws std::bad_alloc instead of ov::Exception; post-fix it must throw ov::Exception. TODO: replace with read_model of a crafted IR whose declared Const shape exceeds its backing data, since a direct Shape-based construction does not model the declared-vs-actual mismatch.

## Suggested fix
Add an explicit upper-bound check in `allocate_buffer` before passing to `AlignedBuffer`: e.g., `constexpr size_t MAX_CONSTANT_BYTES = 1ULL << 30; // 1 GB limit` followed by `OPENVINO_ASSERT(*byte_size <= MAX_CONSTANT_BYTES, "Constant allocation size ", *byte_size, " exceeds maximum allowed");`. Alternatively, enforce the limit in `visit_attributes` after reading `m_element_type`/`m_shape` but before calling `allocate_buffer`, rejecting shapes whose computed byte size exceeds a configurable threshold.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #331.
