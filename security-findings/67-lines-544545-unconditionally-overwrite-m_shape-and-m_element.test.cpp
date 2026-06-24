// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for constant.cpp:544-552: when alloc_buffer_on_visit_attributes(false)
// and a deserialized 'shape'/'element_type' inflate shape_size beyond the existing
// SharedBuffer, Constant::visit_attributes must reject the model instead of leaving
// m_shape larger than m_data->size() (which later makes cast_vector/get_value_strings
// read past the buffer -> CWE-125, detectable under ASan as heap-buffer-overflow).
//
// Target: ov_core_unit_tests   (src/core/tests). Place near existing constant tests
// (e.g. src/core/tests/constant.cpp). Verify exact include paths/helpers against that file.
//
// TODO(symbols): confirm the visitor base class name and on_adapter signatures by reading
//   src/core/tests/visitors/* or openvino/core/attribute_visitor.hpp before relying on this.
#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"

using namespace ov;

// Minimal visitor that injects an attacker-controlled (inflated) shape/element_type
// during visit_attributes, mimicking the IR mmap deserialization path.
namespace {
class InflatingVisitor : public AttributeVisitor {
public:
    explicit InflatingVisitor(std::shared_ptr<AlignedBuffer> small_buf) : m_buf(std::move(small_buf)) {}
    // TODO(symbols): match the real virtual overloads; these names are illustrative.
    void on_adapter(const std::string& name, ValueAccessor<void>& a) override {
        // For the "value" attribute, hand back the small (16-byte) shared buffer.
        if (name == "value") {
            if (auto* sb = dynamic_cast<DirectValueAccessor<std::shared_ptr<AlignedBuffer>>*>(&a))
                sb->set(m_buf);
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<element::Type>& a) override {
        if (name == "element_type") a.set(element::f32);
    }
    void on_adapter(const std::string& name, ValueAccessor<PartialShape>& /*a*/) override {}
    // TODO: supply the Shape adapter override used by Constant to push Shape{4, 1000000}.
private:
    std::shared_ptr<AlignedBuffer> m_buf;
};
}

TEST(constant, visit_attributes_rejects_shape_larger_than_buffer) {
    // Backing store holds only 16 bytes (shape [4] f32) but visitor will inflate shape.
    auto small = std::make_shared<SharedBuffer<std::shared_ptr<void>>>(
        /*ptr*/ reinterpret_cast<char*>(std::malloc(16)), /*size*/ 16, /*so*/ nullptr);
    auto c = std::make_shared<op::v0::Constant>(element::f32, Shape{4}, small->get_ptr(), nullptr);
    c->alloc_buffer_on_visit_attributes(false);  // mmap/IR path -> no reallocation

    InflatingVisitor v(small);
    // Post-fix: visit_attributes must throw because required bytes (4*1000000*4)
    // exceed m_data->size() (16). Pre-fix it returns true and a later cast_vector
    // reads 4,000,000 floats out of bounds (ASan: heap-buffer-overflow READ).
    EXPECT_THROW(c->visit_attributes(v), ov::Exception);
}
