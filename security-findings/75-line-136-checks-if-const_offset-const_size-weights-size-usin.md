# Security finding #75: Line 136 checks `if (const_offset + const_size > weights->size())` …

**Summary:** Line 136 checks `if (const_offset + const_size > weights->size())` …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** OOB heap read: `ov::op::v0::Constant::create` will read `const_size` bytes starting from an attacker-controlled far-OOB address. On most platforms this immediately crashes the process (DoS), but on systems where the memory is mapped the read can expose heap contents across process memory, constituting an information leak. Affects any application loading an attacker-supplied `.xml`/`.bin` IR model through the OpenVINO IR frontend.
**Affected location:** `targets/openvino/src/frontends/ir/src/input_model.cpp:136` — `parse_pre_process()`
**Validated for repos:** openvino
**Trust boundary:** IR XML file (attacker-controlled) → `offset` and `size` attributes on the `<mean>` element, parsed by `get_uint64_attr` at lines 126–127 with no pre-validation, directly fed into uint64_t arithmetic at line 136

## Description / Root cause
Line 136 checks `if (const_offset + const_size > weights->size())` using plain `uint64_t` addition with no pre-overflow guard. If an attacker supplies `const_offset = UINT64_MAX - (const_size - 1)` (e.g. `const_offset = UINT64_MAX - 1935` when `const_size = 1936`), the addition wraps to a small value (≤ `weights->size()`), silently passing the bounds check. At line 185, the unchecked `const_offset` is cast to `size_t` (preserving the huge value on 64-bit), and at line 186 is added to `weights->get_ptr<char>()`, producing a pointer far outside the allocated buffer. This pointer is passed directly to `ov::op::v0::Constant::create` at line 187.

**Validator analysis:** CWE-190 is accurate: line 136 is the only bounds gate and it uses a wrapping uint64_t addition. const_size is pinned at line 128 to shape_size(mean_shape)*input_type.size() (attacker-shapeable to a small, valid value), while const_offset is fully attacker-controlled from the XML `offset` attribute via get_uint64_attr (line 127) with no prior validation. Setting const_offset = UINT64_MAX - const_size + 1 wraps `const_offset + const_size` to 0, which is <= weights->size(), so the OPENVINO_THROW at 137 is skipped. The huge offset is stored (line 139), retrieved (line 185), cast to size_t with no truncation on 64-bit, added to the weights base pointer (line 186) and dereferenced for const_size bytes inside Constant::create (line 187) — a genuine OOB heap read reachable purely from an attacker-supplied IR .xml/.bin, so impact (DoS crash, potential info leak) is accurate. The proposed fix is correct and sufficient: `if (const_size > weights->size() || const_offset > weights->size() - const_size)` is overflow-safe because both operands are pre-bounded before any subtraction (const_size>weights->size() is rejected first, so weights->size()-const_size cannot underflow). The extra standalone `const_offset >= weights->size()` check is redundant once the subtraction form is used but harmless. Note line 128 already constrains const_size, so only the offset path is the real attack surface — the fix correctly targets it. Verdict: openvino validated; openvinoEp rejected (out of reach of the EP's ONNX-only entry).

## Exploit / Proof of Concept
Craft an IR XML whose `<pre-process>` block targets a model with a known input shape (e.g. NCHW [1,3,48,48], giving `const_size = 1*48*48*4 = 9216`). Set `<mean offset="18446744073709542400" size="9216"/>` (where `18446744073709542400 = UINT64_MAX - 9215`). The addition `18446744073709542400 + 9216 = 0` (wraps), passing the guard at line 136. The offset `18446744073709542400` is stored in the `mean_values` set and retrieved at line 185; line 186 computes `weights->get_ptr<char>() + 0xFFFFFFFFFFFF9400`, and line 187 reads 9216 bytes from that address, crashing or leaking data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 integer overflow in
//   openvino/src/frontends/ir/src/input_model.cpp:136 (parse_pre_process)
// Pre-fix: `const_offset + const_size > weights->size()` wraps in uint64_t when
//   offset = UINT64_MAX - const_size + 1, bypassing the bounds check, leading to an
//   OOB pointer at lines 185-187 (caught by ASan as a heap-buffer-overflow read).
// Post-fix (subtraction-form guard): the crafted offset is rejected and the IR
//   frontend throws ov::Exception ("mean value offset and size are out of weights size range").
//
// Harness: ov_ir_frontend_tests (gtest + ASan), IR frontend test tree
//          openvino/src/frontends/ir/tests/.
//
// TODO(symbols): confirm the exact helper used by the existing IR frontend tests to
//   build a model from an in-memory xml string + weights buffer. The IR tests in
//   src/frontends/ir/tests typically use ov::Core::read_model(xml_string, weights_tensor)
//   or a FrontEndManager load_impl helper -- read the neighbouring test .cpp before use.
// TODO(shape): pick input/mean shapes so that shape_size(mean_shape)*type.size() == kConstSize
//   to satisfy the size-match guard at input_model.cpp:128 (here NCHW [1,3,2,2], f32 -> 16 bytes/chan).

#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"

TEST(IRFrontendPreProcessMean, MeanOffsetIntegerOverflowIsRejected) {
    // const_size for one channel of [1,1,2,2] f32 = 4 elems * 4 bytes = 16.
    constexpr uint64_t kConstSize = 16;
    // Wrapping offset: const_offset + const_size overflows uint64_t back to 0.
    constexpr uint64_t kOverflowOffset = 0xFFFFFFFFFFFFFFF0ull; // UINT64_MAX - 16 + 1

    const std::string xml = R"V0G0N(
<net name="overflow_mean" version="11">
  <pre-process reference-layer-name="in">
    <channel id="0"><mean offset=")V0G0N" + std::to_string(kOverflowOffset) +
        R"V0G0N(" size="16"/></channel>
  </pre-process>
  <layers>
    <layer id="0" name="in" type="Parameter" version="opset1">
      <data shape="1,1,2,2" element_type="f32"/>
      <output><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim></port></output>
    </layer>
    <layer id="1" name="res" type="Result" version="opset1">
      <input><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim></port></input>
    </layer>
  </layers>
  <edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/></edges>
</net>)V0G0N";

    // Small weights buffer -- a correct check must reject the wrapping offset, NOT
    // pass it through to weights->get_ptr<char>() + kOverflowOffset.
    ov::Tensor weights(ov::element::u8, ov::Shape{64});

    ov::Core core;
    // Pre-fix: ASan heap-buffer-overflow READ inside Constant::create (input_model.cpp:187).
    // Post-fix: deterministic ov::Exception from the bounds guard (input_model.cpp:136-137).
    EXPECT_THROW(core.read_model(xml, weights), ov::Exception);
}
```
**Build / run:** Build target: cmake --build . --target ov_ir_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_ir_frontend_tests --gtest_filter=IRFrontendPreProcessMean.MeanOffsetIntegerOverflowIsRejected . Expected pre-fix: AddressSanitizer reports 'heap-buffer-overflow READ' originating from ov::op::v0::Constant::create via parse_pre_process (input_model.cpp:186-187), or a crash; the EXPECT_THROW fails because no ov::Exception is raised. Expected post-fix: the subtraction-form bounds check at input_model.cpp:136 throws ov::Exception ('mean value offset and size are out of weights size range') and the test passes. NOTE: verify the read_model(xml, weights) overload / IR test helper name against the existing files in openvino/src/frontends/ir/tests before committing.

## Suggested fix
Replace the single-addition check at line 136 with a pre-subtraction (overflow-safe) form: `if (const_size > weights->size() || const_offset > weights->size() - const_size)`. This avoids the wrapping addition entirely. Additionally, add an independent upper-bound check on `const_offset` alone: `if (const_offset >= weights->size())` before the combined check, to catch large-offset / small-size cases separately.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #75.
