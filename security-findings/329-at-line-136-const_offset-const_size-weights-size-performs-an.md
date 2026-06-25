# Security finding #329: At line 136, `const_offset + const_size > weights->size()` performs…

**Summary:** At line 136, `const_offset + const_size > weights->size()` performs…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound → CWE-125: Out-of-bounds Read
**Severity / Impact:** OOB read of arbitrary memory beyond the weights AlignedBuffer. On a 64-bit system the pointer addition wraps the virtual address space; in practice this causes a segfault/crash (DoS) when the mapped region ends, or an information leak if the wrapped address happens to alias a readable mapping. Any caller loading an attacker-crafted IR model (e.g., via Core::read_model) is affected without any authentication or privilege requirement.
**Affected location:** `targets/openvino/src/frontends/ir/src/input_model.cpp:136` — `anonymous-namespace::parse_pre_process()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied IR XML model file → pugixml attribute parse → get_uint64_attr → unchecked uint64_t addition in bounds guard

## Description / Root cause
At line 136, `const_offset + const_size > weights->size()` performs an unchecked uint64_t addition. If `const_offset = UINT64_MAX - const_size + 1`, the sum wraps to 0, which is not greater than `weights->size()`, silently bypassing the only bounds guard. The secondary check at line 128 only constrains `const_size` (must equal `shape_size(mean_shape) * input_type.size()`), but leaves `const_offset` completely unconstrained. The overflowed `const_offset` is then stored (line 139) and retrieved at line 185 as `const size_t offset`, which at line 186 is used as `weights->get_ptr<char>() + offset` — a pointer arithmetically far outside the AlignedBuffer. Line 187 passes this pointer to `ov::op::v0::Constant::create`, reading `const_size` bytes from attacker-controlled, out-of-bounds memory.

**Validator analysis:** Confirmed by reading input_model.cpp:114-198. const_size and const_offset both come from get_uint64_attr (uint64_t). Line 128 forces const_size == shape_size(mean_shape)*input_type.size() but never constrains const_offset. Line 136's `const_offset + const_size > weights->size()` is unchecked uint64_t addition; setting const_offset = UINT64_MAX - const_size + 1 wraps the sum to 0, which is not > weights->size(), bypassing the only guard. The offset is stored (139), retrieved as size_t (185), added to the weights base pointer (186), and read for const_size bytes by Constant::create (187) — a genuine CWE-190→CWE-125 OOB read. The vulnType/impact are accurate: on 64-bit this yields a far-out-of-bounds pointer → crash/DoS or potential info leak when loading a crafted IR via Core::read_model. The proposedFix is correct and sufficient: the rewrite `const_offset > weights->size() || const_size > weights->size() - const_offset` avoids any wrapping addition and fully closes the hole (size_t/uint64_t widths match weights->size()). Reachable from OpenVINO core (Core::read_model on an IR .xml), hence validated for openvino; not reachable through the ORT OpenVINO EP which uses the ONNX frontend, hence rejected there.

## Exploit / Proof of Concept
Craft an IR XML `<pre-process>` block with a `<mean>` element whose `size` attribute equals `shape_size(mean_shape) * element_size` for the model's input (satisfying the line-128 check) and whose `offset` attribute is set to `UINT64_MAX - const_size + 1`. The addition `(UINT64_MAX - const_size + 1) + const_size` wraps to 0 in uint64_t arithmetic; `0 > weights->size()` is false; the guard at line 136 passes. At line 185-186, `offset` = `UINT64_MAX - const_size + 1` is added to the weights base pointer, producing a pointer that is ~18 EB past the buffer start. Constant::create at line 187 then reads `const_size` bytes from that address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for input_model.cpp:136 (anonymous parse_pre_process).
// Unchecked uint64_t addition `const_offset + const_size > weights->size()` wraps to 0
// when const_offset = UINT64_MAX - const_size + 1, bypassing the bounds guard and causing
// `weights->get_ptr<char>() + offset` (line 186) to point ~18 EB out of bounds.
// Pre-fix: ASan heap-buffer-overflow / segfault (or silent OOB) during read_model.
// Post-fix (overflow-safe check): read_model throws before any pointer arithmetic.
//
// TODO: confirm target name is `ov_ir_frontend_tests` and the helper for reading a model
//       from an in-memory string + weights Tensor matches this repo's IR frontend test tree
//       (e.g. src/frontends/ir/tests/). Adjust input shape/type so that
//       shape_size(mean_shape)*element_size == CONST_SIZE to satisfy the line-128 check.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"

TEST(IRFrontendPreProcess, MeanOffsetIntegerOverflowIsRejected) {
    // input: f32, shape [1,1,1,1] => element_size=4, shape_size(mean_shape)=1 => const_size must be 4
    constexpr uint64_t CONST_SIZE = 4;
    // offset chosen so offset + size wraps to 0 in uint64_t
    const uint64_t bad_offset = UINT64_MAX - CONST_SIZE + 1ULL; // == 0xFFFFFFFFFFFFFFFC

    std::string model = R"V0G0N(<?xml version="1.0"?>
<net name="overflow" version="10">
  <layers>
    <layer id="0" name="in" type="Parameter" version="opset1">
      <data shape="1,1,1,1" element_type="f32"/>
      <pre-process reference-layer-name="in">
        <channel id="0"><mean size=")V0G0N" + std::to_string(CONST_SIZE) +
        R"V0G0N(" offset=")V0G0N" + std::to_string(bad_offset) +
        R"V0G0N("/></channel>
      </pre-process>
      <output><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>1</dim><dim>1</dim></port></output>
    </layer>
    <layer id="1" name="res" type="Result" version="opset1">
      <input><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>1</dim><dim>1</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
  </edges>
</net>)V0G0N";

    // Small weights buffer; any in-bounds offset would need to be < weights size.
    ov::Tensor weights(ov::element::u8, ov::Shape{CONST_SIZE});
    std::memset(weights.data(), 0, CONST_SIZE);

    ov::Core core;
    // Pre-fix the wrapped guard passes and OOB pointer math runs; post-fix this must throw.
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
```
**Build / run:** Build target: ov_ir_frontend_tests (verify name in src/frontends/ir/tests/). Run: ov_ir_frontend_tests --gtest_filter=IRFrontendPreProcess.MeanOffsetIntegerOverflowIsRejected . Expected pre-fix under ASan: heap-buffer-overflow / SEGV inside parse_pre_process at input_model.cpp:186-187 (or test failure because no exception is thrown); post-fix: passes because read_model throws ov::Exception at the overflow-safe bounds check.

## Suggested fix
Replace the single additive comparison at line 136 with two separate, overflow-safe checks:
```cpp
// Before (vulnerable):
if (const_offset + const_size > weights->size()) { ... }

// After (safe):
if (const_offset > weights->size() || const_size > weights->size() - const_offset) {
    OPENVINO_THROW("mean value offset and size are out of weights size range");
}
```
This is idiomatic C++ overflow-safe bounds checking: verify `const_offset` is in range first, then verify that `const_size` fits in the remaining space without performing any addition that could wrap.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #329.
