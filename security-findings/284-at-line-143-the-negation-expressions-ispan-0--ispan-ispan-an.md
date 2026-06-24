# Security finding #284: At line 143, the negation expressions `iSpan < 0 ? -iSpan : iSpan` …

**Summary:** At line 143, the negation expressions `iSpan < 0 ? -iSpan : iSpan` …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** If iSpan==INT_MIN (achieved by start=0, limit=INT_MIN or vice versa), `-iSpan` wraps to INT_MIN, which after `static_cast<size_t>` becomes a huge allocation size passed to `redefineOutputMemory` (line 155), causing heap exhaustion (DoS) or a failed allocation crash. If iStep==INT_MIN, similar overflow supplies 0 or INT_MIN to `div_up`'s divisor, combining with the divide-by-zero risk.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/range.cpp:143` — `Range::getWorkAmount()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX/OV model-supplied scalar RANGE_DELTA or RANGE_LIMIT/RANGE_START input entering CPU EP node execution

## Description / Root cause
At line 143, the negation expressions `iSpan < 0 ? -iSpan : iSpan` and `iStep < 0 ? -iStep : iStep` perform signed negation on `int` values that are directly derived from attacker-controlled inputs (lines 135–141). If `iSpan` or `iStep` equals `INT_MIN` (-2147483648), the negation `-iSpan` or `-iStep` overflows (signed integer overflow is undefined behaviour in C++), potentially producing a negative or unexpected value that is then cast to `size_t` (a large positive number) and used to allocate/iterate over output memory.

**Validator analysis:** Confirmed. At range.cpp:138-143 the i32 path computes span=*stopPtr-*startPtr then `iSpan<0?-iSpan:iSpan` and `iStep<0?-iStep:iStep`. Model-supplied scalars (lines 135-137, read via getSrcDataAtPortAs from RANGE_START/LIMIT/DELTA) are fully attacker-controlled, so iSpan/iStep can be INT_MIN; negating INT_MIN is signed integer overflow (UB), and the subtraction at line 138 (e.g. stop=INT_MAX,start=INT_MIN) is itself an additional overflow point. The (negative) div_up result, cast to size_t at line 143, becomes an enormous work_amount; in the dynamic case it is passed to redefineOutputMemory (line 154-155) as the output extent and to the parallel_nt loop, causing a huge allocation (DoS/OOM) or wraparound iteration. CWE-190 and the DoS/heap-exhaustion impact are accurate. The finding's separate step==0 divide-by-zero is real too (div_up divisor can be 0). The proposed fix (widen to int64_t before computing |span| and |step|) is the right direction and removes the negation UB, but to be fully sufficient it must ALSO compute span in int64 (cast *startPtr/*stopPtr to int64 before subtracting, since line 138 overflows independently) and guard step==0 before div_up. So: correct but incomplete unless the subtraction widening and zero-step check are both added.

## Exploit / Proof of Concept
Supply a Range node with i32 start=0 and limit=INT_MIN (i.e. -2147483648). Then `iSpan = INT_MIN - 0 = INT_MIN`. At line 143, `iSpan < 0` is true, so `-iSpan` is evaluated — this is signed overflow (UB), practically wrapping to INT_MIN on x86. `static_cast<size_t>(INT_MIN)` yields 2147483648 (or 9223372036854775808 on 64-bit), which is passed to `redefineOutputMemory` as the output shape, triggering an enormous allocation attempt.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for openvino/src/plugins/intel_cpu/src/nodes/range.cpp:138-143
// (Range::getWorkAmount). Pre-fix: i32 Range with start=0, limit=INT_MIN, step=1
// triggers signed-overflow UB on `-iSpan` and a div_up result that casts to an
// enormous size_t work_amount -> redefineOutputMemory huge allocation (ASan/OOM).
// Post-fix: int64 magnitude computation + zero-step guard must reject/clamp so no
// absurd output extent is produced.
//
// SKELETON: exact single-op Range test fixture symbols must be copied from the
// intel_cpu single-layer test tree before this compiles.
#include <gtest/gtest.h>
// TODO: include the correct headers, e.g.
//   #include "single_layer_tests/classes/range.hpp" or the ngraph op + CPU
//   reference infra under src/plugins/intel_cpu/tests/unit/ and shared_test_classes.

TEST(CpuRangeNode, IntMinSpanDoesNotOverflowWorkAmount) {
    // TODO: build an i32 Range op with constant inputs start=0, limit=INT_MIN, step=1
    //   using the same builder helper the existing Range tests use
    //   (ov::op::v4::Range or the CPU node wrapper).
    // TODO: compile for CPU as a DYNAMIC-shape model so redefineOutputMemory() path
    //   (range.cpp:153-155) is exercised.
    // EXPECTATION (post-fix): inference either throws a bounded ov::Exception or
    //   produces a sane (non-astronomical) output extent; it must NOT attempt a
    //   ~2^31/2^63 element allocation.
    // ASSERT_NO_FATAL_FAILURE(infer());  // pre-fix: ASan signed-overflow / bad_alloc
    // EXPECT_LT(output_element_count, kSaneUpperBound);
    GTEST_SKIP() << "TODO: wire up intel_cpu Range single-op fixture symbols";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (or the intel_cpu single-layer test binary that owns Range). Run: ov_cpu_unit_tests --gtest_filter='CpuRangeNode.IntMinSpanDoesNotOverflowWorkAmount'. Pre-fix expectation: ASan 'signed integer overflow: -(-2147483648)' at range.cpp:143 and/or std::bad_alloc from redefineOutputMemory due to size_t-cast huge work_amount. Post-fix: passes (bounded extent / thrown ov::Exception).

## Suggested fix
Use `uint32_t` (or `int64_t`) arithmetic for the span and step magnitude computation before calling `div_up`. For example: `auto uSpan = static_cast<uint32_t>(iSpan < 0 ? (iSpan == INT_MIN ? static_cast<int64_t>(INT_MIN) * -1 : -iSpan) : iSpan);` or cast both to `int64_t` before negating: `int64_t s64 = iSpan; int64_t st64 = iStep; return static_cast<size_t>(div_up(s64 < 0 ? -s64 : s64, st64 < 0 ? -st64 : st64));`. Also add the non-zero step check described in the first finding.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #284.
