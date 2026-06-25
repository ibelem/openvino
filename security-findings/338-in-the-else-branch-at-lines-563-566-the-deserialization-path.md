# Security finding #338: In the `else` branch at lines 563-566 (the deserialization path whe…

**Summary:** In the `else` branch at lines 563-566 (the deserialization path whe…

**CWE IDs:** CWE-20: Improper Input Validation / CWE-125: Out-of-bounds Read
**Severity / Impact:** If an attacker provides a `shape` declaring N elements but a `value` blob encoding only M < N strings, `m_data` holds M `std::string` objects while `m_shape` claims N. Every subsequent read using `shape_size(m_shape)` as the element count reads past the end of the `StringAlignedBuffer`. Concretely, `evaluate()` (line 594-597) does `std::copy_n(src_strings, num_elements, dst_strings)` where `num_elements = shape_size(m_shape) = N`; reading N objects from an M-object buffer causes an OOB read of heap-resident `std::string` objects (potentially reading adjacent heap metadata or other strings). `get_value_strings()` (line 445) passes `shape_size(m_shape)` as `num_elements` into the same unsafe iterator path. Crash (DoS) is near-certain; info leak or attacker-controlled pointer dereference is plausible if heap layout is controlled.
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:566` — `Constant::visit_attributes / Constant::evaluate()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-supplied XML `shape` attribute + serialized `value` blob → StringAlignedBuffer element count vs. shape_size(m_shape)

## Description / Root cause
In the `else` branch at lines 563-566 (the deserialization path where `m_data` is neither a `StringAlignedBuffer` nor a `SharedBuffer<Tensor>`), a local null `string_aligned_buffer` is populated by `visitor.on_attribute("value", string_aligned_buffer)` — whose element count is determined solely by the serialized value blob — and then directly assigned to `m_data` (`m_data = string_aligned_buffer;`) with no check that `string_aligned_buffer->get_num_elements() == shape_size(m_shape)`. The shape value is already committed to `m_shape` from attacker input at line 545, and `m_alloc_buffer_on_visit_attributes` may be false (or `need_to_reallocate` false), so `allocate_buffer` may not run at all. No post-assignment validation reconciles the two counts.

**Validator analysis:** The defect is real for the OpenVINO core. In Constant::visit_attributes (constant.cpp:540-573) m_shape is taken from the XML 'shape' attribute (line 545) while, in the deserialization else-branch (562-566), the string buffer's element count is determined solely by the packed-blob header (aux_unpack_string_tensor strings_count, string_aligned_buffer.cpp:31-55) and assigned to m_data with no check that get_num_elements()==shape_size(m_shape). validate_and_infer_types (536-538) only sets the output type and performs no size reconciliation. Constant::evaluate (586-603) and get_value_strings then use shape_size(m_shape) as num_elements in std::copy_n over a possibly smaller std::string buffer — a genuine heap OOB read of std::string objects (CWE-125 / CWE-20). vulnType and impact (DoS near-certain, info-leak plausible) are accurate. The unpack routine already validates the blob's *internal* consistency (header size, offsets) but NOT the cross-attribute shape/count agreement, which is exactly the missing check. The proposed fix (OPENVINO_ASSERT on count equality after line 566, plus a size check for all string paths in validate_and_infer_types) is correct and sufficient; I would put the canonical check in validate_and_infer_types for element::string so the SharedBuffer and dynamic_pointer_cast branches (555,557) are covered too, since a mutating visitor could also desync those. For openvinoEp the path is not reachable: the ORT OpenVINO EP ingests ONNX graphs (handled by the onnx frontend / normal Constant constructors) or its own compiled cache blob, not free-form attacker IR string Constants, so the visit_attributes deserialization else-branch is never driven from the EP boundary.

## Exploit / Proof of Concept
Supply a crafted OpenVINO XML/blob model where the Constant node's `element_type='string'`, `shape='[1 1000]'` (1000 elements), and the `value` blob contains a packed-string payload representing only 1 string. During deserialization with `m_alloc_buffer_on_visit_attributes=false` (or when `need_to_reallocate` is false), the code takes the `else` branch at line 563, the visitor parses the blob and creates a 1-element `StringAlignedBuffer`, and `m_data = string_aligned_buffer` stores it without validation. Any call to `evaluate()` then calls `std::copy_n(src_strings, 1000, dst_strings)`, reading 999 `std::string` objects past the single valid object in the buffer.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for constant.cpp:564-566 (Constant::visit_attributes string deserialization else-branch).
// Pre-fix: a Constant whose 'shape' attribute claims N string elements but whose deserialized
//          StringAlignedBuffer holds only M<N strings is accepted; Constant::evaluate (constant.cpp:594-597)
//          then std::copy_n(src, N, dst) reads past the M-element buffer -> ASan heap-buffer-overflow.
// Post-fix: the count mismatch is rejected at visit_attributes / validate_and_infer_types via OPENVINO_ASSERT.
//
// NOTE (skeleton): triggering the exact else-branch requires a mock AttributeVisitor that (a) writes
// m_shape = {1, 1000} for "shape" and (b) installs a 1-element StringAlignedBuffer for "value".
// The concrete visitor symbol and on_attribute(StringAlignedBuffer&) overload must be copied from the
// real IR deserializer test fixtures before this compiles.
#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

using namespace ov;

namespace {
// TODO: replace with the project's real deserialization-style AttributeVisitor (see IR frontend tests).
class MismatchedStringVisitor : public AttributeVisitor {
public:
    // TODO: implement on_adapter overloads required by the base class.
    void on_adapter(const std::string&, ValueAccessor<void>&) override {}
    // TODO: override the shape and StringAlignedBuffer on_attribute hooks to inject
    //       shape = Shape{1,1000} but a StringAlignedBuffer with get_num_elements()==1.
};
}  // namespace

TEST(op_eval_string_constant, deserialized_count_must_match_shape) {
    // Construct an empty string Constant put into deserialization mode.
    auto c = std::make_shared<op::v0::Constant>(element::string, Shape{0});
    c->alloc_buffer_on_visit_attributes(false);  // forces the else-branch at constant.cpp:562

    MismatchedStringVisitor visitor;  // declares shape [1,1000] but only 1 string in the value blob
    // The fix must reject the mismatch here (or in validate_and_infer_types).
    EXPECT_THROW(c->visit_attributes(visitor), ov::Exception);

    // Defense-in-depth: even if visit_attributes were lenient, evaluate must not OOB-read.
    // TensorVector outs;
    // EXPECT_THROW(c->evaluate(outs, {}), ov::Exception);  // pre-fix: ASan heap-buffer-overflow in copy_n
}
```
**Build / run:** Build target: ov_core_unit_tests (cmake --build . --target ov_core_unit_tests). Run: ov_core_unit_tests --gtest_filter='op_eval_string_constant.deserialized_count_must_match_shape'. With -DENABLE_SANITIZER=ON the pre-fix binary reports 'AddressSanitizer: heap-buffer-overflow READ' in std::copy_n at constant.cpp:597 (reading std::string objects past the 1-element StringAlignedBuffer); post-fix the OPENVINO_ASSERT throws ov::Exception and the EXPECT_THROW passes. NOTE: the MismatchedStringVisitor mock and StringAlignedBuffer overload are TODO placeholders — copy the real visitor from the IR deserializer test fixtures before building.

## Suggested fix
After line 566, add an explicit element-count assertion before `m_data` is assigned: `OPENVINO_ASSERT(string_aligned_buffer && string_aligned_buffer->get_num_elements() == shape_size(m_shape), "Deserialized string buffer element count (", string_aligned_buffer ? string_aligned_buffer->get_num_elements() : 0, ") does not match shape element count (", shape_size(m_shape), ")");`. Additionally, after `visit_attributes` completes, `validate_and_infer_types()` (or a dedicated size-check helper) should verify `m_data->get_num_elements() == shape_size(m_shape)` for string-type constants to guard all paths (including the first `dynamic_pointer_cast` branch where visitor mutation could also produce a mismatch).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #338.
