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