# Security finding #511: The bounds check `if (const_offset + const_size > weights->size())`…

**Summary:** The bounds check `if (const_offset + const_size > weights->size())`…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** At line 186, `weights->get_ptr<char>() + offset` dereferences a pointer calculated from the unchecked `const_offset` value. Because the bounds check was bypassed, `offset` can point arbitrarily far past the end of the weights allocation. The subsequent `ov::op::v0::Constant::create` call at line 187 reads `const_size` bytes from that address, yielding an out-of-bounds read across an arbitrarily large region of process memory — enabling information disclosure of heap/stack contents and potentially a crash (DoS) during model load in any application that parses untrusted OpenVINO IR files.
**Affected location:** `targets/openvino/src/frontends/ir/src/input_model.cpp:136` — `parse_pre_process()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted XML <pre-process> element attributes `offset` and `size` from an attacker-supplied IR model file, parsed as `uint64_t` at lines 126–127

## Description / Root cause
The bounds check `if (const_offset + const_size > weights->size())` at line 136 performs unsigned 64-bit addition of two attacker-controlled values. If the adversary supplies e.g. `const_offset = UINT64_MAX - 50` and `const_size = 100`, the sum wraps to 49, which is less than `weights->size()`, so the check silently passes even though both values individually exceed the buffer. No overflow pre-check (`__builtin_add_overflow`, split check, or `const_size > weights->size() || const_offset > weights->size() - const_size`) is present.

**Validator analysis:** Confirmed in openvino. At lines 126-127 both `const_size` and `const_offset` are parsed as uint64_t from untrusted IR XML <mean> attributes. Line 128 constrains `const_size` to equal `shape_size(mean_shape) * input_type.size()`, but the attacker controls `mean_shape` so a small const_size (e.g. 100) is achievable, and `const_offset` is entirely unconstrained. The check at line 136 adds them in uint64 space; choosing const_offset = UINT64_MAX-50 with const_size=100 wraps the sum to 49 < weights->size(), bypassing the guard. Line 185 assigns offset=const_offset (size_t, 64-bit) and line 186 computes an out-of-bounds pointer that Constant::create reads const_size bytes from (line 187) — a genuine OOB read / DoS during untrusted IR model load. CWE-190 -> OOB read is accurate; impact (info disclosure / crash) is accurate. The proposed fix (`const_size > weights->size() || const_offset > weights->size() - const_size`) is correct and overflow-safe: it guards const_size first making the subtraction non-wrapping, then bounds const_offset. Sufficient. For openvinoEp the defect file is not in the EP and the IR pre-process path is not reached from the EP's ONNX-model boundary, so rejected there.

## Exploit / Proof of Concept
Craft an IR model `.xml` file whose `<mean>` element inside `<pre-process>` carries `offset="18446744073709551565"` (UINT64_MAX-50) and `size="100"`. The parser reads both as `uint64_t` at lines 126–127. The sum at line 136 overflows to 49; if `weights->size() >= 49` (trivially satisfied by a minimal `.bin`), the check passes. At line 185 `offset` receives UINT64_MAX-50 and line 186 computes a pointer ~18 exabytes past the beginning of the weights buffer, which is then read for 100 bytes — causing an OOB read or SIGSEGV.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for CWE-190 integer overflow at
// openvino/src/frontends/ir/src/input_model.cpp:136 (parse_pre_process).
// Pre-fix: const_offset + const_size wraps, bypassing the bounds check, and
// line 186 builds an OOB pointer that Constant::create reads -> ASan heap-buffer-overflow / SIGSEGV.
// Post-fix (split check): the IR frontend rejects the model with ov::Exception.
//
// TODO: this needs a crafted IR fixture; emitted as a skeleton.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(IRFrontend, PreProcessMeanOffsetIntegerOverflowRejected) {
    ov::Core core;

    // TODO: build a minimal IR <net> whose input feeds a <pre-process> with a
    // <channel><mean size="4" offset="18446744073709551613"/></channel> so that
    // (offset + size) wraps below weights->size(). size must equal
    // shape_size(mean_shape)*input_type.size() to pass the line 128 check.
    const std::string ir_xml = R"V0G0N(
        <!-- TODO: full IR graph with <pre-process><channel id="0"><mean size="4" offset="18446744073709551613"/></channel></pre-process> -->
    )V0G0N";

    // TODO: weights buffer large enough that the *wrapped* sum (e.g. 1) <= size,
    // but the real offset is far out of bounds. Use a Tensor of a few bytes.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});

    // Pre-fix: read_model proceeds and triggers an OOB read building the mean Constant.
    // Post-fix: the overflow-safe split check throws.
    EXPECT_THROW(core.read_model(ir_xml, weights), ov::Exception);
}
```
**Build / run:** Build target: ov_ir_frontend_tests (the IR frontend's gtest target under src/frontends/ir/tests). Run: ov_ir_frontend_tests --gtest_filter='IRFrontend.PreProcessMeanOffsetIntegerOverflowRejected'. Pre-fix expectation: ASan reports heap-buffer-overflow (or SIGSEGV) inside Constant::create reading weights->get_ptr<char>()+offset at input_model.cpp:186-187. Post-fix expectation: read_model throws ov::Exception ('mean value offset and size are out of weights size range') and the test passes. TODO: supply the crafted IR XML and matching weights Tensor fixture before this compiles/runs.

## Suggested fix
Replace the single addition-based check at line 136 with a split, overflow-safe check:
```cpp
if (const_size > weights->size() || const_offset > weights->size() - const_size) {
    OPENVINO_THROW("mean value offset and size are out of weights size range");
}
```
This first guards `const_size <= weights->size()` (making the subtraction safe), then verifies `const_offset <= weights->size() - const_size`, eliminating the unsigned wrap-around.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #511.
