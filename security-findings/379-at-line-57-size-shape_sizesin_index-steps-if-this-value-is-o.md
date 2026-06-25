# Security finding #379: At line 57, `size = shape_sizes[in_index] / steps`. If this value i…

**Summary:** At line 57, `size = shape_sizes[in_index] / steps`. If this value i…

**CWE IDs:** CWE-197: Numeric Truncation Error
**Severity / Impact:** A model with an i4/u4 Concat where `shape_sizes[in_index] / steps` is odd causes the output tensor to contain uninitialized heap data (CWE-908) for the truncated nibble positions. This can expose stale heap contents to callers reading the output tensor, constituting an information disclosure. validate_and_infer_types() performs no check that per-step element counts are even for nibble types.
**Affected location:** `targets/openvino/src/core/reference/src/op/concat.cpp:60` — `concat()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted model inputs with i4/u4 element type enter via Concat::evaluate(); no validation enforces even element counts per step before calling reference::concat()

## Description / Root cause
At line 57, `size = shape_sizes[in_index] / steps`. If this value is odd (e.g., 3 elements per step), `size /= 2` at line 60 truncates to 1 via integer division, discarding the final nibble. `copy_func` then copies 1 byte instead of the required ceil(3/2)=2 bytes. The `out_offset` at line 63 advances by 1, so the second byte of that output region is never written — it retains whatever was in the heap-allocated output buffer.

**Validator analysis:** The truncation is real in the openvino core reference: for u4/i4, `size /= 2` floors, so a per-step element count of 3 yields 1 copied byte instead of ceil(3/2)=2. The trailing output byte is never written, so the freshly set_shape()'d output Tensor retains uninitialized heap content (CWE-908) AND produces incorrect concatenation results — the CWE-197/CWE-908 categorization is accurate. Concat::evaluate() (and ConstantFolding via evaluate) reaches this with no even-count validation, so it is reachable within openvino. The proposed fix is WRONG/dangerous: replacing `size /= 2` with `(size+1)/2` over-copies — e.g. two [3] u4 inputs into a [6] (3-byte) output would write 2+2=4 bytes, a heap buffer OVERFLOW (worse than the leak), and out_offset would then exceed the buffer. The only sound fix is to reject non-byte-aligned per-step nibble segments: add `OPENVINO_ASSERT((shape_sizes[in_index] / steps) % 2 == 0, "i4/u4 concat: per-step element count must be even")` at the top of the inner loop (or validate in Concat::validate_and_infer_types), because nibble-packed concat across non-byte-aligned boundaries cannot be done with plain memcpy at all. The reachability from the EP is not established; only openvino is validated.

## Exploit / Proof of Concept
Craft a model with a u4 Concat of two inputs each shaped [3] on axis=0: steps=1, N=3, size=3/1=3, then size/=2 → 1. copy_func copies 1 byte into output instead of 2. The output Tensor's second byte is uninitialized heap memory. Any downstream consumer reading the 3-element result reads one nibble of stale heap data for the final element.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for openvino/src/core/reference/src/op/concat.cpp:57-63
// (CWE-197 truncation -> CWE-908 uninitialized output byte for odd u4 per-step counts).
//
// Pre-fix: concat() silently copies floor(3/2)=1 byte per input and leaves the
//          trailing output byte unwritten -> ASan/MSan flags use-of-uninitialized,
//          and the result is wrong, but evaluate() does NOT throw.
// Post-fix: an even-count guard in reference::concat()/Concat::validate_and_infer_types()
//          rejects the non-byte-aligned u4 segment, so evaluate() throws.
//
// Lives alongside the core op_eval tests (target: ov_core_unit_tests, e.g. src/core/tests/eval.cpp).
// TODO: confirm exact include paths / test file location against the local core tests tree.
#include "gtest/gtest.h"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(eval_concat, u4_odd_per_step_count_must_be_rejected) {
    // Two u4 inputs of shape [3] concatenated on axis 0 -> per-step element count = 3 (odd).
    // 3 nibbles occupy 2 bytes; reference::concat() truncates 3/2 -> 1 byte (concat.cpp:57-60).
    auto a = std::make_shared<op::v0::Constant>(element::u4, Shape{3}, std::vector<uint8_t>{0, 0});
    auto b = std::make_shared<op::v0::Constant>(element::u4, Shape{3}, std::vector<uint8_t>{0, 0});
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{a, b}, 0);

    Tensor in_a(element::u4, Shape{3});
    Tensor in_b(element::u4, Shape{3});
    Tensor out(element::u4, Shape{6});  // 3 bytes; pre-fix byte[2] stays uninitialized
    TensorVector inputs{in_a, in_b};
    TensorVector outputs{out};

    // Once the even-count guard is added, the malformed u4 segment is rejected up-front.
    // Pre-fix this returns true and leaves out.data()[2] uninitialized (caught by ASan/MSan).
    EXPECT_ANY_THROW(concat->evaluate(outputs, inputs));
}
```
**Build / run:** Build target: ov_core_unit_tests. Run: ov_core_unit_tests --gtest_filter='eval_concat.u4_odd_per_step_count_must_be_rejected'. Pre-fix, evaluate() returns true and (under -fsanitize=memory or MSan/Valgrind) reports 'use-of-uninitialized-value' on the unwritten trailing output byte; the EXPECT_ANY_THROW also fails because no guard exists. Post-fix (even-count OPENVINO_ASSERT in reference::concat or Concat::validate_and_infer_types) evaluate() throws ov::Exception and the test passes with no sanitizer report.

## Suggested fix
Use ceiling division when converting element count to byte count for nibble types: replace `size /= 2` with `size = (size + 1) / 2`. Additionally, add a shape validation check in Concat::validate_and_infer_types() (or at the top of reference::concat()) asserting that for i4/u4 element types, each per-step element count is even, to catch malformed models early: `OPENVINO_ASSERT((shape_sizes[in_index] / steps) % 2 == 0, "i4/u4 concat: per-step element count must be even")`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #379.
