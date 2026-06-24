# Security finding #179: For the string branch (lines 594-597), `num_elements = shape_size(m…

**Summary:** For the string branch (lines 594-597), `num_elements = shape_size(m…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Type-confusion/UB crash or potential information leak when std::string copy constructor is invoked on garbage memory. Affects string-type Constants in shared-buffer deserialization contexts (e.g., models with embedded string tensors).
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:594` — `Constant::evaluate()`
**Validated for repos:** openvino
**Trust boundary:** Deserialization of attacker-supplied IR model into a shared-buffer string Constant (m_alloc_buffer_on_visit_attributes=false)

## Description / Root cause
For the string branch (lines 594-597), `num_elements = shape_size(m_shape)` is computed from the possibly attacker-inflated `m_shape`, and `std::copy_n(src_strings, num_elements, dst_strings)` accesses `num_elements` std::string objects starting at `m_data->get_ptr()`. If m_shape was inflated without reallocation in visit_attributes (line 549 guard), src_strings is indexed past the end of the actual StringAlignedBuffer, reading junk bytes as std::string objects and invoking their copy constructors on corrupted data — undefined behavior that can lead to crashes or arbitrary reads.

**Validator analysis:** The vuln type (CWE-125 OOB read leading to UB on std::string copy-construction) is accurate for the string branch. In Constant::evaluate (constant.cpp:593-597) num_elements is derived from m_shape, which during shared-buffer IR deserialization is read from XML at visit_attributes:545, while m_data (a StringAlignedBuffer) is populated independently at lines 555-566 with no check that shape_size(m_shape) equals the buffer's element count. The guarded reallocation at line 549 is skipped when m_alloc_buffer_on_visit_attributes=false, so an inflated XML shape over a smaller .bin string buffer yields std::copy_n reading std::string objects past the buffer end — genuine UB/OOB read. Note the validating constructor Constant(type,shape,AlignedBuffer) at 301-313 DOES enforce *constant_size==data_size and would reject the mismatch, but the string deserialization path bypasses it via the AttributeVisitor, so the gap is real. The proposed fix is directionally correct but should prefer StringAlignedBuffer::get_num_elements() (see string_aligned_buffer.cpp:95) over m_data->size()/sizeof(std::string), which assumes element::string.size()==sizeof(std::string); a cleaner fix adds the OPENVINO_ASSERT(num_elements<=string_buffer->get_num_elements()) check, or better, validates shape vs buffer once in visit_attributes right after line 566 so all consumers (not just evaluate) are protected. Not reachable from the ONNX-Runtime EP boundary, hence openvinoEp rejected.

## Exploit / Proof of Concept
Craft an IR where a string-type Constant node's XML shape claims more elements than the .bin StringAlignedBuffer actually contains, using a shared-buffer Constant (m_alloc_buffer_on_visit_attributes=false). evaluate() at line 594 computes num_elements from the inflated m_shape and calls std::copy_n with that count, indexing std::string objects past the end of m_data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for constant.cpp:594-597 (Constant::evaluate string branch OOB read).
// Pre-fix: with m_shape inflated past the StringAlignedBuffer element count via the
// AttributeVisitor deserialization path, evaluate() std::copy_n's past the buffer end
// (ASan heap-buffer-overflow / UB on std::string copy ctor).
// Post-fix: evaluate()/visit_attributes rejects the shape/data mismatch via OPENVINO_ASSERT.
//
// Harness: ov_core_tests (gtest). Place near openvino/src/core/tests/constant.cpp.
#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

using namespace ov;

namespace {
// Minimal visitor that injects a SMALL StringAlignedBuffer (1 element) but reports a
// LARGER shape so shape_size(m_shape) > buffer element count.
class MismatchVisitor : public AttributeVisitor {
public:
    void on_adapter(const std::string&, ValueAccessor<void>&) override {}
    // TODO: override the adapters actually used by Constant::visit_attributes:
    //   - element_type adapter -> set element::string
    //   - shape adapter        -> set Shape{4}   (inflated)
    //   - "value" adapter (std::shared_ptr<ov::StringAlignedBuffer>&) -> assign a buffer
    //     constructed for ONLY 1 element: std::make_shared<StringAlignedBuffer>(1, element::string.size()*1, 64, true)
    // The exact ValueAccessor<T> specializations must be learned by reading
    // openvino/src/core/src/op/constant.cpp visit_attributes (lines 540-573) and the
    // AttributeAdapter for StringAlignedBuffer.
};
}  // namespace

TEST(constant_string_oob, evaluate_rejects_shape_buffer_mismatch) {
    // TODO: instantiate a default string Constant, call alloc_buffer_on_visit_attributes(false),
    //       drive visit_attributes(MismatchVisitor) to set inflated shape + 1-element buffer.
    op::v0::Constant c{};  // TODO: construct via the deserialization-style path
    c.alloc_buffer_on_visit_attributes(false);
    MismatchVisitor v;
    c.visit_attributes(v);

    ov::TensorVector outputs;
    // Pre-fix this triggers an OOB read inside std::copy_n at constant.cpp:597.
    // Post-fix the size check must throw before the copy.
    EXPECT_THROW(c.evaluate(outputs, {}), ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_core_tests. Run: ./ov_core_tests --gtest_filter=constant_string_oob.* . Expected pre-fix under ASan: 'heap-buffer-overflow READ' in op::v0::Constant::evaluate (std::copy_n over std::string objects, constant.cpp:597). Post-fix: test passes because the added shape-vs-StringAlignedBuffer element-count assertion throws ov::Exception before the copy.

## Suggested fix
Before std::copy_n, verify that num_elements does not exceed the actual element count stored in m_data: `const size_t actual_elements = std::dynamic_pointer_cast<ov::StringAlignedBuffer>(m_data) ? m_data->size() / sizeof(std::string) : 0; OPENVINO_ASSERT(num_elements <= actual_elements, "String Constant shape/data mismatch"); std::copy_n(src_strings, num_elements, dst_strings);`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #179.
