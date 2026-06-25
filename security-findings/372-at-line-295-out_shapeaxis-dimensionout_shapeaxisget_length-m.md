# Security finding #372: At line 295, `out_shape[axis] = Dimension{out_shape[axis].get_lengt…

**Summary:** At line 295, `out_shape[axis] = Dimension{out_shape[axis].get_lengt…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Incorrect (negative or near-zero) output shape is inferred for a ConcatOutputDescription scan output. Downstream consumers that allocate a buffer based on this corrupted shape will either allocate a near-zero or wrong-size buffer, leading to out-of-bounds writes (CWE-787) when the actual loop output data is written into it, or to silent model mis-inference.
**Affected location:** `targets/openvino/src/core/src/op/loop.cpp:295` — `Loop::validate_and_infer_types()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Attacker-controlled trip_count constant value (int64_t) stored in m_num_iterations and used in output shape dimension arithmetic

## Description / Root cause
At line 295, `out_shape[axis] = Dimension{out_shape[axis].get_length() * m_num_iterations}` multiplies two int64_t values — the per-iteration axis length from the body and `m_num_iterations` — without overflow protection. `get_length()` returns `int64_t`, and `m_num_iterations` is `int64_t`. If an attacker supplies a trip_count that folds to a large positive int64_t (e.g., 2^60) and the body produces a slice size of 4, the product 4 × 2^60 = 2^62 overflows int64_t on the next multiplication of a similarly-large value. With a trip_count of INT64_MAX/2 + 1 and a slice size ≥ 2, the product wraps to a negative int64_t Dimension, corrupting the inferred output shape.

**Validator analysis:** The defect is real and reachable. At loop.cpp:118-125 the TripCount constant is read and, when the body condition folds to true (condition_always_true), m_num_iterations is set directly to the attacker value val[0] with NO range/sign validation. At loop.cpp:294-295, for a ConcatOutputDescription with a static concat axis and m_num_iterations != -1, the code computes get_length()*m_num_iterations as a raw int64_t multiply. A large positive trip_count (e.g. INT64_MAX/2+1) times a per-iteration slice size >=2 overflows signed int64_t (UB), or a negative trip_count yields a negative product. CWE-190 is the correct primary classification. The downstream CWE-787 impact is partly speculative — a wrapped-negative value passed to Dimension{}/Interval is more likely to be clamped or to throw later than to produce a tiny buffer that is then overflowed — but the integer-overflow defect itself stands. The proposed fix's claim that 'a prior fix at line 124 rejects negative/excessive trip_count' is false: no such check exists today. The checked-multiply OPENVINO_ASSERT is good but insufficient: its `m_num_iterations <= 0 || ...` short-circuit lets negative trip_counts through and still produces a negative dimension. Better fix: at line 124 add NODE_VALIDATION_CHECK(this, val[0] >= 0, "TripCount must be non-negative") (and reject absurdly large values), AND wrap the line-295 multiply with an overflow check that throws via NODE_VALIDATION_CHECK rather than silently wrapping.

## Exploit / Proof of Concept
Set the trip_count Constant of a Loop with a ConcatOutputDescription to INT64_MAX/2 + 1 (= 4611686018427387904). The body produces a 1-D tensor of size 2 along the concatenation axis. At line 295, `2 * 4611686018427387904 = 9223372036854775808` overflows signed int64_t (UB/wrap) to −9223372036854775808, setting the output dimension to a large negative Dimension. When this shape is later used to allocate a runtime tensor and data is written into it, a heap buffer overflow occurs.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 at openvino/src/core/src/op/loop.cpp:295 (multiply
// out_shape[axis].get_length() * m_num_iterations with no overflow guard) and the
// missing bounds check at loop.cpp:124 (m_num_iterations = val[0]).
// Pre-fix: validate_and_infer_types() computes a wrapped/negative concat-axis
// dimension (UB signed overflow) instead of rejecting the input.
// Post-fix: a non-negative-but-overflowing trip_count must be rejected via
// ov::Exception (NODE_VALIDATION_CHECK).
//
// Harness: ov_core_unit_tests, type_prop style (src/core/tests/type_prop/loop.cpp).
// TODO: confirm exact helper/include names by reading src/core/tests/type_prop/loop.cpp
//       and src/core/tests/type_prop/tensor_iterator.cpp before use.

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

TEST(type_prop, loop_concat_output_tripcount_overflow_is_rejected) {
    // ---- Build a trivial body: Xi (slice) + cond(true) ----
    auto Xi = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});
    auto cur_iter = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1});
    // body execution condition folds to constant true -> condition_always_true
    auto body_cond = op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto body_out  = std::make_shared<op::v0::Result>(Xi);
    auto body_cond_res = std::make_shared<op::v0::Result>(body_cond);

    auto body = std::make_shared<Model>(ResultVector{body_out, body_cond_res},
                                        ParameterVector{cur_iter, Xi});

    // ---- Loop external inputs ----
    // trip_count chosen so slice_size(2) * trip_count overflows int64_t.
    const int64_t kEvilTrip = 4611686018427387904LL; // INT64_MAX/2 + 1
    auto trip_count = op::v0::Constant::create(element::i64, Shape{1}, {kEvilTrip});
    auto exec_cond  = op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto X          = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});

    auto loop = std::make_shared<op::v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(op::v5::Loop::SpecialBodyPorts{0, 1});
    loop->set_invariant_input(Xi, X);
    // ConcatOutputDescription along axis 0 -> triggers loop.cpp:295 multiply.
    loop->get_concatenated_slices(body_out, 0, 1, 1, -1, 0);

    // Pre-fix: validate_and_infer_types() silently wraps to a negative dim (UB).
    // Post-fix: the overflowing/over-large trip_count must be rejected.
    EXPECT_THROW(loop->validate_and_infer_types(), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests. Run: ./ov_core_unit_tests --gtest_filter=type_prop.loop_concat_output_tripcount_overflow_is_rejected . Pre-fix expectation: with -fsanitize=undefined the multiply at loop.cpp:295 reports 'signed integer overflow: 2 * 4611686018427387904 cannot be represented in type long' and the test FAILS (no exception thrown). Post-fix: NODE_VALIDATION_CHECK throws ov::Exception and the test PASSES. TODO: verify op::v5::Loop helper signatures (get_concatenated_slices, set_special_body_ports) against src/core/tests/type_prop/loop.cpp before committing.

## Suggested fix
Guard the multiplication with a checked arithmetic helper before assigning: e.g., `int64_t slice_len = out_shape[axis].get_length(); OPENVINO_ASSERT(m_num_iterations <= 0 || slice_len <= INT64_MAX / m_num_iterations, "TripCount * slice_size overflows int64_t"); out_shape[axis] = Dimension{slice_len * m_num_iterations};`. A prior fix at line 124 (rejecting negative/excessive trip_count) also prevents this path from being reached with attacker-chosen values.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #372.
