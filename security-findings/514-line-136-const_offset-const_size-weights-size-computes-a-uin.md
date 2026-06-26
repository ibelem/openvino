# Security finding #514: Line 136: `const_offset + const_size > weights->size()` computes a …

**Summary:** Line 136: `const_offset + const_size > weights->size()` computes a …

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** The computed raw pointer `data = weights->get_ptr<char>() + const_offset` (line 186) points entirely outside the mapped weights region. The Constant created at line 187 then calls `std::memcpy` from that invalid address (constant.cpp:297), causing a crash (SIGSEGV / DoS) or, on systems with adjacent mappings, an out-of-bounds read of weight data (CWE-125), potentially leaking sensitive model weights or enabling heap-layout attacks via controlled read length.
**Affected location:** `targets/openvino/src/frontends/ir/src/input_model.cpp:136` — `parse_pre_process()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied .xml pre-process element `<mean offset=... size=...>` attributes → weights AlignedBuffer pointer used in Constant construction

## Description / Root cause
Line 136: `const_offset + const_size > weights->size()` computes a uint64_t addition of two attacker-controlled values without overflow protection. If the sum wraps around (> UINT64_MAX), the check evaluates to false and is silently bypassed, even when `const_offset` is far beyond the end of the weights buffer.

**Validator analysis:** The integer-overflow claim is accurate. const_size is constrained by the equality check at line 128 (must equal shape_size(mean_shape)*input_type.size()), but const_offset (line 127, get_uint64_attr) is fully attacker-controlled and never independently bounds-checked. At line 136 the sum const_offset+const_size is a uint64_t addition: choosing a huge const_offset (e.g. ~UINT64_MAX-615) plus a small valid const_size wraps below weights->size(), bypassing the throw. Line 185-187 then use the raw const_offset to form `weights->get_ptr<char>() + offset` and Constant::create copies const_size bytes from that pointer (constant.cpp memcpy), giving an OOB read/SIGSEGV — CWE-190 leading to CWE-125/DoS, as stated. Impact characterization is correct. The proposed fix `if (const_size > weights->size() || const_offset > weights->size() - const_size)` is overflow-safe and sufficient (the const_size>size() guard prevents the size-t underflow in `size()-const_size`). One refinement: keep it after the line 128 size-match check (already present) — that constrains const_size but not offset, so the offset guard is the essential addition. For the chain, only the OpenVINO repo's IR-frontend boundary reaches this; the ONNX EP does not parse IR XML, so it is rejected.

## Exploit / Proof of Concept
Craft a model XML with a 4D NCHW input parameter (e.g., shape [1,3,H,W]) and a pre-process block containing `<mean offset='18446744073709551000' size='N'/>` where N = shape_size({1,H,W}) * sizeof(float) (e.g., H=W=512 → N=3145728). Since N < 615, offset+N wraps to < weights->size(), the check at line 136 passes. Line 186 then computes `weights->get_ptr<char>() + 18446744073709551000`, a pointer far outside any valid mapping; the subsequent `Constant::create` → `memcpy` from that pointer crashes the process or reads attacker-chosen OOB memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190 in openvino/src/frontends/ir/src/input_model.cpp:136
// (parse_pre_process). Pre-fix: a crafted <mean offset=HUGE size=N> where
// offset+size wraps past UINT64_MAX bypasses the `const_offset + const_size >
// weights->size()` check, so line 186 forms an out-of-bounds pointer and
// Constant::create memcpy's from it (ASan heap-buffer-overflow / SIGSEGV).
// Post-fix (offset checked against weights->size()-const_size): convert_model
// must throw ov::Exception instead.
//
// Goes in the IR frontend gtest target (ov_ir_frontend_tests), e.g.
// src/frontends/ir/tests/frontend_test_basic.cpp style.
#include <gtest/gtest.h>
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/core/except.hpp"

using namespace ov;

TEST(IRFrontendPreProcess, MeanOffsetOverflowRejected) {
    // TODO: build a minimal IR <net> with a 4D NCHW Parameter (shape [1,3,H,W])
    //       and a <pre-process><channel id=..><mean size='N' offset='OFF'/> block
    //       for every channel, where:
    //         N   = shape_size({1,H,W}) * sizeof(float)   (passes line 128)
    //         OFF = 18446744073709551000ULL                (so OFF + N wraps < N)
    //       Pick H=W small so N is tiny and the wrapped sum < weights size.
    std::string xml = R"(<!-- TODO: crafted IR XML as described above --> )";

    // TODO: allocate a weights AlignedBuffer larger than the wrapped sum so the
    //       bypassed check would otherwise pass; e.g. shared_ptr<AlignedBuffer>.
    auto weights = std::make_shared<ov::AlignedBuffer>(/*size=*/4096);

    ov::frontend::FrontEndManager fem;
    auto fe = fem.load_by_framework("ir");
    ASSERT_NE(fe, nullptr);
    // TODO: load via the IR FE's stream+weights input variant used by the
    //       existing ov_ir_frontend_tests fixtures.
    auto in_model = fe->load(/* TODO: stream(xml) + weights */);
    // Expect rejection of the overflowing offset rather than an OOB read.
    EXPECT_THROW(fe->convert(in_model), ov::Exception);
}
```
**Build / run:** Build target: ov_ir_frontend_tests. Run: ov_ir_frontend_tests --gtest_filter=IRFrontendPreProcess.MeanOffsetOverflowRejected . With ASan, the pre-fix code reports a heap-buffer-overflow / SEGV inside ov::op::v0::Constant::create -> std::memcpy (constant.cpp ~297) reached from input_model.cpp:186-187; post-fix the load throws ov::Exception ("mean value offset and size are out of weights size range") and the test passes.

## Suggested fix
Replace the plain addition with an overflow-safe check before line 136: `if (const_size > weights->size() || const_offset > weights->size() - const_size)`. This avoids unsigned wrap-around entirely. Also add a similar guard for the `const_offset` alone to reject obviously invalid values.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #514.
