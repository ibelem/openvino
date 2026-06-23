# Security finding #8: At lines 819-820, `offset` and `size` are `size_t` values read from…

**Summary:** At lines 819-820, `offset` and `size` are `size_t` values read from…

**CWE IDs:** CWE-190: Integer Overflow / CWE-125: Out-of-bounds Read
**Severity / Impact:** Same as finding 1: out-of-bounds read beyond the `.bin` weights buffer. Can cause process crash (DoS) or information disclosure of heap/stack memory under attacker-controlled model load.
**Affected location:** `targets/openvino/src/core/xml_util/src/xml_deserialize_util.cpp:828` — `XmlDeserializer::on_adapter (StringAlignedBuffer branch)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied IR XML file ingested by OpenVINO model-load (IR frontend loader)

## Description / Root cause
At lines 819-820, `offset` and `size` are `size_t` values read from untrusted XML attributes. The guard at line 828 (`if (m_weights->size() < offset + size) OPENVINO_THROW(...)`) evaluates `offset + size` in `size_t`. With `offset = SIZE_MAX` and `size = 1`, the sum wraps to 0; `m_weights->size() < 0` is never true for an unsigned comparison, so the throw is never reached. Line 830 then computes `m_weights->get_ptr<char>() + offset` with `offset = SIZE_MAX`, yielding an OOB pointer passed directly into `unpack_string_tensor`.

**Validator analysis:** The flaw is real. At lines 819-820 `offset` and `size` are read unchecked from untrusted XML attributes into size_t. The sole guard at line 828 evaluates `offset + size` in unsigned size_t arithmetic; with offset=SIZE_MAX (0xFFFFFFFFFFFFFFFF) and size=1 the sum wraps to 0, so `m_weights->size() < 0` is always false and the OPENVINO_THROW never fires. Line 830 then computes an out-of-bounds base pointer that is handed to unpack_string_tensor (line 832). The vuln type (CWE-190 leading to CWE-125 OOB read) and impact (DoS / heap info disclosure on model load) are accurate. The proposed fix is correct and sufficient: replacing the additive check with `if (offset > m_weights->size() || size > m_weights->size() - offset) OPENVINO_THROW(...)` removes the overflow-prone addition while preserving the bound; subtraction `size() - offset` is safe because it is guarded by the prior `offset > size()` test. Factoring a shared validated-offset helper (also used by set_constant_num_buffer, which has the same offset+size pattern) is a good hardening recommendation. Note the analogous additive checks for the AlignedBuffer numeric-const branch should be audited with the same fix. Reachability from the OVEP repo is not established — the EP path consumes ONNX, not IR XML, so only the openvino core repo is validated.

## Exploit / Proof of Concept
Craft an IR XML node with type `Const` and `<data element_type="string" shape="1" offset="18446744073709551615" size="1"/>`. When the `StringAlignedBuffer` adapter branch processes this: `offset + size` wraps to 0, the inequality at line 828 is false, and `m_weights->get_ptr<char>() + 0xFFFFFFFFFFFFFFFF` is passed as `data` to `unpack_string_tensor` at line 832.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for integer-overflow OOB read in
//   openvino/src/core/xml_util/src/xml_deserialize_util.cpp:828-830
// Pre-fix: offset=SIZE_MAX, size=1 makes `offset + size` wrap to 0, the
//   `m_weights->size() < offset+size` guard is bypassed, and
//   get_ptr<char>()+SIZE_MAX is read by unpack_string_tensor -> ASan OOB.
// Post-fix (split check `offset > size() || size > size()-offset`):
//   the malformed Const is rejected with ov::Exception instead.
//
// SKELETON: building a minimal valid IR XML + matching .bin/weights tensor
// that reaches the StringAlignedBuffer 'value'/'Const' branch requires a
// crafted fixture; the exact opset/string-const schema must be copied from a
// known-good string-Const IR. TODOs below name what is missing.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(IRFrontend, string_const_offset_overflow_is_rejected) {
    // TODO: replace with a minimal IR whose op type="Const" carries
    //   <data element_type="string" shape="1"
    //         offset="18446744073709551615" size="1"/>
    //   so on_adapter() takes the StringAlignedBuffer branch (line 812-833).
    //   Confirm exact attribute/opset names against an existing string-Const
    //   IR test fixture under src/frontends/ir/tests.
    const std::string model = R"V0G0N(
<net name="overflow" version="11">
    <layers>
        <!-- TODO: parameter + string Const(offset=SIZE_MAX,size=1) + result -->
    </layers>
    <edges/>
</net>
)V0G0N";

    // Non-empty weights so m_weights is set (line 826) but small enough that a
    // real SIZE_MAX offset is wildly out of bounds.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});

    ov::Core core;
    // Pre-fix: ASan heap-buffer-overflow inside unpack_string_tensor.
    // Post-fix: clean throw of "Incorrect weights in bin file!".
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
```
**Build / run:** Build target: ov_ir_frontend_tests (the IR frontend unit test target that exercises xml_deserialize_util; confirm exact name from src/frontends/ir/tests/CMakeLists.txt). Run: ov_ir_frontend_tests --gtest_filter=IRFrontend.string_const_offset_overflow_is_rejected. Build with -DENABLE_SANITIZER=ON. Expected pre-fix failure: AddressSanitizer 'heap-buffer-overflow READ' originating from XmlDeserializer::on_adapter -> unpack_string_tensor (xml_deserialize_util.cpp:830-832). Post-fix: test passes (ov::Exception thrown by the corrected split bounds check at line 828).

## Suggested fix
Replace the single overflow-susceptible comparison with a two-part check: `if (offset > m_weights->size() || size > m_weights->size() - offset) OPENVINO_THROW("Incorrect weights in bin file!");`. This avoids the addition entirely. Alternatively, factor out a shared validated-offset helper (reusable by both `set_constant_num_buffer` and this branch) that performs the same split check.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #8.
