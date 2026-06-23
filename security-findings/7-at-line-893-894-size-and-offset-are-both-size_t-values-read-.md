# Security finding #7: At line 893-894, `size` and `offset` are both `size_t` values read …

**Summary:** At line 893-894, `size` and `offset` are both `size_t` values read …

**CWE IDs:** CWE-190: Integer Overflow / CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds read / memory corruption. Any attacker who can supply a crafted `.xml` model file (local user, cloud inference endpoint receiving a user-uploaded model) can cause the inference engine to read arbitrary process memory beyond the weights buffer, potentially leaking secrets or triggering a crash/DoS.
**Affected location:** `targets/openvino/src/core/xml_util/src/xml_deserialize_util.cpp:895` — `XmlDeserializer::set_constant_num_buffer()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied IR XML file ingested by OpenVINO model-load (IR frontend loader)

## Description / Root cause
At line 893-894, `size` and `offset` are both `size_t` values read directly from untrusted XML attributes via `pugixml::get_uint64_attr`. The bounds check at line 895 (`m_weights->size() >= offset + size`) evaluates `offset + size` in `size_t` arithmetic. If an attacker supplies `offset = SIZE_MAX` (0xFFFFFFFFFFFFFFFF) and `size = 1`, the addition wraps to 0, which is always less than or equal to `m_weights->size()`, so the OPENVINO_ASSERT passes. Line 897 then computes `m_weights->get_ptr<char>() + offset` with `offset = SIZE_MAX`, resulting in a pointer far beyond the allocated buffer, and the data is subsequently consumed by `unpack_string_tensor` or used to construct a `SharedBuffer`.

**Validator analysis:** The flaw is real in openvino: at lines 893-895 `size` and `offset` are size_t values taken verbatim from XML via pugixml::get_uint64_attr with no individual bounds clamp, and the combined check `m_weights->size() >= offset + size` performs the addition in size_t, so an overflowing pair (offset=0xFFFFFFFFFFFFFFFF, size=1) wraps to 0 and defeats the guard; the subsequent `m_weights->get_ptr<char>() + offset` and use in unpack_string_tensor / SharedBuffer reads out of bounds. The vulnType (CWE-190 enabling CWE-125 OOB read) and the impact (info-leak / crash on crafted IR model load) are accurate; this is reachable from core read_model when the IR frontend is selected. The proposed fix is correct and sufficient: replacing the single-expression check with `OPENVINO_ASSERT(offset <= m_weights->size() && size <= m_weights->size() - offset, "Incorrect weights in bin file!")` eliminates the wraparound (both terms are non-negative, subtraction cannot underflow because offset<=size()). The alternative widened-arithmetic form also works but on 64-bit platforms size_t==uint64_t so the cast adds nothing; the subtraction-based form is the clean fix. For the EP repo the verdict is rejected only because the path is not reachable there, not because the defect is unreal.

## Exploit / Proof of Concept
Craft an IR XML Const node with `<data element_type="i8" shape="1" offset="18446744073709551615" size="1"/>` paired with a valid (even empty) `.bin` weights file. When `set_constant_num_buffer` is called: `offset = 0xFFFFFFFFFFFFFFFF`, `size = 1`, `offset + size = 0` (wraps), assertion `m_weights->size() >= 0` passes, and `m_weights->get_ptr<char>() + 0xFFFFFFFFFFFFFFFF` accesses memory at an address ~18 EB past the buffer base.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190/CWE-125 in
//   openvino/src/core/xml_util/src/xml_deserialize_util.cpp:893-895
//   (XmlDeserializer::set_constant_num_buffer)
// Pre-fix: offset=0xFFFFFFFFFFFFFFFF + size=1 wraps to 0, the bounds
//   OPENVINO_ASSERT at line 895 passes, then get_ptr<char>()+offset (line 897)
//   is dereferenced out-of-bounds -> ASan heap-buffer-overflow / read.
// Post-fix: the input is rejected and read_model throws ov::Exception.
//
// Harness: IR frontend tests (ov_core_unit_tests / IR frontend test target).
// TODO: confirm the exact target & include path by reading the nearest
//       existing IR-frontend test dir (e.g. src/frontends/ir/tests/) and
//       mirror its read_model(model_str, weights_tensor) helper.
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

TEST(IRFrontendOverflow, ConstOffsetSizeWraparoundRejected) {
    // Const with offset = SIZE_MAX, size = 1; offset+size wraps to 0 pre-fix.
    std::string model = R"V0G0N(<?xml version="1.0"?>
<net name="oob" version="11">
    <layers>
        <layer id="0" name="c" type="Const" version="opset1">
            <data element_type="i8" shape="1" offset="18446744073709551615" size="1"/>
            <output>
                <port id="0" precision="I8"><dim>1</dim></port>
            </output>
        </layer>
        <layer id="1" name="r" type="Result" version="opset1">
            <input><port id="0" precision="I8"><dim>1</dim></port></input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>)V0G0N";

    // Tiny, valid weights buffer so m_weights is non-null and small.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});
    std::memset(weights.data(), 0, weights.get_byte_size());

    ov::Core core;
    // TODO: if Core::read_model(std::string, ov::Tensor) overload differs in
    //       this tree, switch to the IR test fixture's createModelFromString().
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests (or the IR frontend test target ov_ir_frontend_tests once confirmed). Run: ./ov_core_unit_tests --gtest_filter=IRFrontendOverflow.ConstOffsetSizeWraparoundRejected . Pre-fix with -DENABLE_SANITIZER=ON expect ASan 'heap-buffer-overflow READ' (or SEGV) at xml_deserialize_util.cpp:897; post-fix the OPENVINO_ASSERT rejects the input and the test passes via the thrown ov::Exception.

## Suggested fix
Before the bounds check, verify that the addition cannot overflow: add `OPENVINO_ASSERT(offset <= m_weights->size() && size <= m_weights->size() - offset, "Incorrect weights in bin file!");` replacing the single-expression check. Alternatively use saturating/widened arithmetic: `OPENVINO_ASSERT(static_cast<uint64_t>(offset) + static_cast<uint64_t>(size) >= offset && m_weights->size() >= offset + size, ...);`. Also consider capping both `offset` and `size` against `m_weights->size()` before any arithmetic.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #7.
