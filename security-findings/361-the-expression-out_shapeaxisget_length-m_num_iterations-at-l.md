# Security finding #361: The expression `out_shape[axis].get_length() * m_num_iterations` at…

**Summary:** The expression `out_shape[axis].get_length() * m_num_iterations` at…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** The overflowed (negative) int64_t is passed into `Dimension{value}` (dimension.cpp:70), which delegates to `Interval(value)` (interval.cpp:61). `Interval::Interval(value_type val)` calls `clip(val)` which is `std::max(0, std::min(s_max, val))` (interval.cpp:10–11), so a negative overflow result is silently clamped to 0. A zero-valued dimension for the concatenation axis propagates through `set_output_type` (line 300) to all downstream consumers. When those consumers allocate output buffers sized by the (zero) inferred shape, but the loop body actually writes `actual_dim × iterations` worth of data, a heap buffer overflow occurs at runtime. Affected parties: any application loading an untrusted ONNX model through OpenVINO.
**Affected location:** `targets/openvino/src/core/src/op/loop.cpp:295` — `Loop::validate_and_infer_types()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX/IR model file → shape-inference engine → downstream buffer allocation

## Description / Root cause
The expression `out_shape[axis].get_length() * m_num_iterations` at line 295 is a plain signed int64_t multiplication with no overflow guard. `m_num_iterations` is assigned verbatim from `val[0]` (cast_vector<int64_t> of a model-supplied constant) at line 124 with no range check beyond `val.size() == 1`. `out_shape[axis].get_length()` is the int64_t body-output dimension, also model-controlled. Both operands can independently be large positive int64_t values. Their product overflows int64_t silently (signed overflow is UB in C++; in practice on x86 it wraps to a negative or near-zero value).

**Validator analysis:** Confirmed by reading loop.cpp:113-125 and 274-300. m_num_iterations is set from the model's TripCount constant (line 124) with no value/range validation (only val.size()==1 at 120-122). At line 295 the product out_shape[axis].get_length() * m_num_iterations is a plain signed int64_t multiply with no overflow guard, and both factors are attacker-controlled — so CWE-190 (signed integer overflow, UB) is real and reachable from an untrusted model. The vulnType is accurate. The IMPACT, however, is partly overstated: dimension.cpp:70-71 + interval.cpp:10-12/61-64 confirm a negative wrap is clamp()ed to 0, so the most reliable consequence is a wrong (zero/clamped or wrapped-positive) inferred dimension, not a guaranteed heap overflow — the claimed runtime heap-overflow assumes the loop actually executes ~4.6e18 iterations, which is itself an unbounded-resource/DoS condition, and the shape mismatch could feed downstream allocators. The defect is genuine regardless. The proposed fix is correct and matches the project's own pattern: OpenVINO already has clip_times() in interval.cpp:14-22 doing exactly this saturating-multiply guard, so the cleanest fix is to route the dimension product through that saturating logic (or the proposed `dim_len > INT64_MAX / m_num_iterations` check setting Dimension::dynamic()), plus validating val[0] >= 0 at line 123. Sufficient as stated, though using/aligning with clip_times is preferable to a hand-rolled check.

## Exploit / Proof of Concept
Craft an ONNX model with a Loop node whose TripCount is a constant scalar set to, e.g., 0x4000000000000001 (≈ 4.6 × 10^18). If the loop body produces an output with an axis dimension of 2, the product 2 × 0x4000000000000001 = 0x8000000000000002, which overflows int64_t to a large negative number. `clip()` clamps it to 0, so `set_output_type` sets the output axis to 0. Downstream code allocates a zero-element buffer for that output. When the loop executes its actual iterations and writes output, it writes past the end of the zero-sized allocation. No check between lines 123–125 and 294–295 prevents this path.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression test for CWE-190 at openvino/src/core/src/op/loop.cpp:295
// (unguarded signed int64_t `out_shape[axis].get_length() * m_num_iterations`).
// Pre-fix: with a huge TripCount constant the product overflows int64_t (UBSan: signed
// integer overflow) and the concat-axis dimension is silently clamped to 0 / wraps.
// Post-fix: the product is saturated (Dimension::dynamic or s_max), so the inferred
// concat-axis dimension is NOT a bogus small/zero static value.
//
// Harness: ov_core_unit_tests (gtest). Place in openvino/src/core/tests/type_prop/loop.cpp
// alongside the existing TYPED/TEST cases for op::v5::Loop.
//
// NOTE: building a full Loop (body model + SpecialBodyPorts + ConcatOutputDescription)
// requires the exact builder helpers used by the existing loop type_prop tests; the
// TODOs below mark the pieces to copy from those tests verbatim. Marked skeleton because
// the precise body-construction symbols were not read here.

#include <gtest/gtest.h>
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

TEST(type_prop_loop, trip_count_concat_axis_overflow_is_saturated) {
    // TODO: construct body Model exactly as existing loop type_prop tests do:
    //   - a body Parameter with a static rank-1 shape (e.g. {2}) feeding the concat output
    //   - a body condition Result that is Constant(true)  -> condition_always_true
    // TODO: build op::v5::Loop, set special body ports (body_condition_output_idx,
    //       current_iteration_input_idx) and get_concatenated_slices(...) for the body value
    //       along axis 0, matching the helper used in the existing tests.

    // Crafted TripCount so that dim(2) * trip_count overflows int64_t and (pre-fix) wraps:
    const int64_t kOverflowTrip = 0x4000000000000001LL;  // 2 * this overflows int64_t
    auto trip_count = op::v0::Constant::create(element::i64, Shape{}, {kOverflowTrip});
    auto exec_cond  = op::v0::Constant::create(element::boolean, Shape{}, {true});

    // TODO: auto loop = make_loop(trip_count, exec_cond, body, /*concat axis*/0, /*dim*/2);
    // loop->validate_and_infer_types();

    // EXPECTED post-fix: the concat-axis dimension must NOT be a wrapped/clamped small static
    // value. A correct saturating multiply yields a dynamic dimension (or s_max), so:
    //   const auto& out_ps = loop->get_output_partial_shape(/*concat out idx*/0);
    //   EXPECT_TRUE(out_ps[0].is_dynamic())
    //       << "loop.cpp:295 must saturate the trip_count*dim product, not overflow to 0";
    // Pre-fix this fails: out_ps[0] is a static 0 (or wrapped value) and UBSan reports
    // signed-integer-overflow at loop.cpp:295.
    GTEST_SKIP() << "skeleton: fill in body/Loop construction from existing loop type_prop tests";
}
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests. Run: ./ov_core_unit_tests --gtest_filter='type_prop_loop.trip_count_concat_axis_overflow_is_saturated'. Build with -fsanitize=undefined; pre-fix UBSan reports 'signed integer overflow: 2 * 4611686018427387905 cannot be represented in type long' at openvino/src/core/src/op/loop.cpp:295 and the concat-axis dim infers as static 0; post-fix the dimension is dynamic and no UBSan diagnostic fires.

## Suggested fix
Before the multiplication, check for overflow: `int64_t dim_len = out_shape[axis].get_length(); if (m_num_iterations > 0 && dim_len > std::numeric_limits<int64_t>::max() / m_num_iterations) { out_shape[axis] = Dimension::dynamic(); } else { out_shape[axis] = Dimension{dim_len * m_num_iterations}; }`. Additionally, validate `val[0] >= 0` at line 123 (after the cast_vector) and reject or clamp trip-count values that exceed a reasonable upper bound (e.g., INT32_MAX or a configurable limit).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #361.
