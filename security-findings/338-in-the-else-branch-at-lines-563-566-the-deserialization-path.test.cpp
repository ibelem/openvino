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
