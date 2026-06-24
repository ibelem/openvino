# Security finding #220: Line 599: `std::memcpy(outputs[0].data(), get_data_ptr(), outputs[0…

**Summary:** Line 599: `std::memcpy(outputs[0].data(), get_data_ptr(), outputs[0…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Out-of-bounds heap read from m_data's backing allocation. Reads heap metadata or adjacent allocations into the output tensor buffer — information disclosure and/or process crash (SIGBUS/SIGSEGV). Reachable at inference time on any model using TypeRelaxed<Constant> (mixed-precision inference) or anywhere a caller supplies a pre-existing output tensor of a wider numeric type.
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:599` — `Constant::evaluate()`
**Validated for repos:** openvino
**Trust boundary:** Inference-time output tensor passed into evaluate() whose element type may differ from the Constant's internal m_element_type (e.g. TypeRelaxed<Constant> path or external caller providing a pre-sized output tensor).

## Description / Root cause
Line 599: `std::memcpy(outputs[0].data(), get_data_ptr(), outputs[0].get_byte_size())` uses the output tensor's independently-computed byte size as the copy length. When `outputs[0]` has a larger element type than the Constant (e.g. f32 vs f16), `outputs[0].get_byte_size() = shape_size(m_shape) * output_type.size()` exceeds `m_data->size() = shape_size(m_shape) * m_element_type.size()`, causing memcpy to read past the end of m_data. The function does NOT guard against this: line 590-591 only resets the shape, not the type, so the size mismatch persists.

**Validator analysis:** The defect is real as code: Constant::evaluate (constant.cpp:586-603) sets only the shape (line 591, set_shape does not alter element type) and then at line 599 copies outputs[0].get_byte_size() bytes from get_data_ptr(). When the supplied output tensor's element type is wider than m_element_type, get_byte_size() = shape_size*output_type.size() exceeds m_data->size() = shape_size*m_element_type.size(), yielding an out-of-bounds heap read — CWE-125 is accurate, and impact (info disclosure / crash) is plausible. The mismatch path is deliberately reachable: evaluate_lower (611-612) and evaluate_upper (619-620) call evaluate() precisely when outputs[0].get_element_type() != m_element_type, with the comment 'for TypeRelaxed<Constant>'. There is no size guard or type check in the else branch. Reachability from the openvino C++/evaluate API and the bound-evaluation framework is sound; reachability from the ONNX Runtime EP is not demonstrated because TypeRelaxed<Constant> is produced by internal transforms, not directly by model input — hence openvinoEp is na. The proposed fix's std::min(get_byte_size(), outputs[0].get_byte_size()) clamp removes the OOB but still produces semantically wrong (raw byte-reinterpreted) data; the better fix is the OPENVINO_ASSERT(outputs[0].get_element_type() == m_element_type, ...) plus a real type-converting path for TypeRelaxed callers — that is correct and sufficient to stop the OOB and surface the misuse.

## Exploit / Proof of Concept
1. Create a Constant with element type f16 and shape [N] so m_data->size() = N*2 bytes. 2. Call evaluate_lower() with a non-empty outputs vector where outputs[0] has element type f32 and the same shape [N]. 3. evaluate_lower() at line 611-612 sees the type mismatch and calls evaluate(outputs, {}). 4. evaluate() at line 591 calls outputs[0].set_shape(m_shape) — this reallocates the output buffer to N*4 bytes but leaves the type as f32. 5. outputs[0].get_byte_size() returns N*4; get_data_ptr() points to m_data of size N*2. 6. memcpy reads N*4 bytes from m_data, over-reading N*2 bytes past the allocation boundary. Achievable from ONNX Runtime EP by loading a model with TypeRelaxed constants and triggering constant folding/evaluate_lower.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
#include <gtest/gtest.h>

#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

// Encodes the fix for the OOB read at
// openvino/src/core/src/op/constant.cpp:599.
//
// Constant stores f16 (2 bytes/elem). The caller supplies a *wider* f32
// (4 bytes/elem) output tensor of the same shape, mimicking the
// TypeRelaxed<Constant> path that evaluate_lower()/evaluate_upper()
// (constant.cpp:611-612 / 619-620) deliberately route into evaluate().
// Line 591 only resets the shape, leaving the f32 element type; line 599
// then memcpy's outputs[0].get_byte_size() (= N*4) bytes out of m_data
// (= N*2 bytes) -> heap-buffer-overflow under ASan.
//
// Pre-fix: ASan reports a read past the constant's backing allocation.
// Post-fix (OPENVINO_ASSERT type-match or size clamp): no over-read; the
// type mismatch is rejected via ov::Exception.
TEST(constant, evaluate_wider_output_type_no_oob_read) {
    const Shape shape{8};
    std::vector<ov::float16> values(shape_size(shape), ov::float16(1.0f));
    auto c = op::v0::Constant::create(element::f16, shape, values);

    ov::TensorVector outputs;
    outputs.emplace_back(element::f32, shape);  // wider than the f16 constant

    // Pre-fix this memcpy over-reads (ASan abort). Post-fix the mismatch
    // must be rejected rather than copied past the source buffer.
    EXPECT_THROW(c->evaluate(outputs, {}), ov::Exception);
}
```
**Build / run:** Build target: ov_core_unit_tests. Run: ./ov_core_unit_tests --gtest_filter=constant.evaluate_wider_output_type_no_oob_read . With an ASan build, the pre-fix code aborts with 'heap-buffer-overflow READ' inside std::memcpy at constant.cpp:599 (reading N*4 bytes from the N*2-byte f16 buffer). After the fix (type-match OPENVINO_ASSERT), the test passes because evaluate() throws ov::Exception instead of over-reading.

## Suggested fix
Replace line 599 with a size-bounded copy using the Constant's own buffer size: `const size_t copy_bytes = std::min(outputs[0].get_byte_size(), get_byte_size()); std::memcpy(outputs[0].data(), get_data_ptr(), copy_bytes);`. Better still, assert that types match before memcpy: `OPENVINO_ASSERT(outputs[0].get_element_type() == m_element_type, "element type mismatch in Constant::evaluate");` — and route TypeRelaxed callers through a proper type-converting path rather than a raw memcpy.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #220.
