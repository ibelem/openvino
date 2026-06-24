# Security finding #67: Lines 544–545 unconditionally overwrite `m_shape` and `m_element_ty…

**Summary:** Lines 544–545 unconditionally overwrite `m_shape` and `m_element_ty…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Any subsequent accessor that derives an iteration count from `shape_size(m_shape)` will read past the end of the real data buffer. Confirmed sinks: (1) `Constant::get_value_strings()` (constant.cpp:445) passes `shape_size(m_shape)` directly as `num_elements` to `ValuesToString::visit`, which calls `std::transform(first, first + num_elements, ...)` over the raw pointer; (2) `cast_vector<T>()` (constant.hpp:249–257) passes `get_num_elements_to_cast(-1)` = `shape_size(m_shape)` (constant.cpp:642) to `data::cast_n` which walks the raw pointer for that many elements; (3) `are_all_data_elements_bitwise_identical()` (constant.cpp:503/509/514/519) uses `shape_size(m_shape)` as the `size` arg to `test_bitwise_identical(data, size)`. All three read attacker-controlled numbers of bytes beyond the buffer boundary. An adversary can leak adjacent mmap'd or heap memory (information disclosure) or trigger a segfault/process crash (DoS). Affected users: any caller that loads an IR model from a binary file using the shared-buffer path.
**Affected location:** `targets/openvino/src/core/src/op/constant.cpp:549` — `Constant::visit_attributes()`
**Validated for repos:** openvino
**Trust boundary:** Deserialized model file attributes → AttributeVisitor::on_attribute("shape", m_shape) and on_attribute("element_type", m_element_type) at lines 544–545

## Description / Root cause
Lines 544–545 unconditionally overwrite `m_shape` and `m_element_type` with attacker-controlled deserialized values, then `need_to_reallocate` is computed at line 547. The only buffer guard is at line 549: `if (m_alloc_buffer_on_visit_attributes && need_to_reallocate)`. When `m_alloc_buffer_on_visit_attributes` is `false` (set via `Constant::alloc_buffer_on_visit_attributes(false)`, used in the IR-binary / mmap loading path), the entire body of the `if` is dead regardless of how large the new shape is. `m_shape` is now the inflated attacker value while `m_data` still points to the original, smaller `SharedBuffer` wrapping the mmap'd binary data. No check anywhere reconciles `shape_size(m_shape)` with `m_data->size()` after this point.

**Validator analysis:** CWE-125 (out-of-bounds read) is the correct category and the impact (info-leak/DoS via accessors that derive element counts from shape_size(m_shape)) is accurate: get_value_strings (445) feeds shape_size into std::transform over get_data_ptr(); get_num_elements_to_cast (642) returns shape_size for cast_vector's data::cast_n; are_all_data_elements_bitwise_identical (503/509/514/519) walks size=shape_size. None of these reconcile shape_size*elem_size against m_data->size(). The eager constructor (constant.cpp:306-313) DOES assert size match, but the default-construct + visit_attributes deserialization path bypasses it, and validate_and_infer_types (536-538) only sets the output shape. The one residual doubt is whether the IR frontend's value-attribute adapter independently rejects size!=shape*elem_size (that file is not present in this checkout, so I could not read it); given the eager ctor's matching assert was deliberately omitted from this path and there is no other reconciliation visible in core, the gap is real. The proposed fix is correct and is the most robust location: add an UNCONDITIONAL check after lines 544-545 comparing get_memory_size_safe(m_element_type,m_shape) against m_data->size() and throw via OPENVINO_ASSERT when the existing buffer is too small (variant (b) is preferable to (a) since variant (a) would reject legitimate no-realloc cases where size already matches). It should run regardless of m_alloc_buffer_on_visit_attributes, after m_data is populated at line 569 (or as a post-condition at end of visit_attributes), so both the alloc and mmap paths are validated.

## Exploit / Proof of Concept
Craft an OpenVINO IR model: keep the binary weights file at its normal size for a constant of shape [4] float32 (16 bytes), but change the XML `shape` attribute to `4 1000000`. OpenVINO's IR reader calls `Constant::alloc_buffer_on_visit_attributes(false)` so that `m_data` will be filled from the mmap'd binary rather than newly allocated. After deserialization completes, `m_shape` = {4,1000000} (4 000 000 elements) while `m_data->size()` = 16. A call to `node->cast_vector<float>()` will compute `num_elements_to_cast = 4000000` (constant.cpp:642) and invoke `data::cast_n(..., get_data_ptr(), 4000000, dst)`, reading 16 000 000 bytes starting from the 16-byte mmap'd region, leaking or crashing on whatever follows in the address space.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build: cmake --build . --target ov_core_unit_tests. Run: ./ov_core_unit_tests --gtest_filter='constant.visit_attributes_rejects_shape_larger_than_buffer'. Pre-fix (or if you skip the throw and call c->cast_vector<float>()) ASan reports 'heap-buffer-overflow READ of size N' in ov::op::v0::data::cast_n / std::transform off the 16-byte SharedBuffer; post-fix the visit_attributes OPENVINO_ASSERT throws ov::Exception and the test passes. NOTE: skeleton — fill in the real AttributeVisitor on_adapter overloads (and the Shape adapter pushing Shape{4,1000000}) from src/core/tests/visitors before building.

## Suggested fix
After the `visitor.on_attribute` calls at lines 544–545, add a validation step that is unconditional on `m_alloc_buffer_on_visit_attributes`. When `need_to_reallocate` is true and `m_alloc_buffer_on_visit_attributes` is false, either (a) reject the model with `OPENVINO_ASSERT(false, "Constant shape/type changed but buffer reallocation is disabled")`, or (b) check that `m_data != nullptr && m_data->size() >= *ov::util::get_memory_size_safe(m_element_type, m_shape)` and throw if the buffer is too small. Concretely, insert before line 554:
```cpp
if (!m_alloc_buffer_on_visit_attributes && need_to_reallocate) {
    const auto required = ov::util::get_memory_size_safe(m_element_type, m_shape);
    OPENVINO_ASSERT(required && m_data && m_data->size() >= *required,
        "Deserialized shape/type requires ", required.value_or(0),
        " bytes but existing buffer holds ", (m_data ? m_data->size() : 0));
}
```


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #67.
