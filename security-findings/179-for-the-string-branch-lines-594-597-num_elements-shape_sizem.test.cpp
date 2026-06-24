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
