// Agent-authored; NOT compiled or run against the source tree — review before use.
#include <gtest/gtest.h>

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

// Encodes the fix for the OOB heap read in Constant evaluation.
// Source of defect:
//   src/core/src/op/constant.cpp:599  -> memcpy(outputs[0].data(), get_data_ptr(), outputs[0].get_byte_size())
//   src/core/src/op/constant.cpp:611-612 -> evaluate_lower() forwards a type-mismatched
//                                            (wider) output tensor straight into evaluate().
// Pre-fix behaviour: a same-shape f32 (4-byte) output tensor causes evaluate() to copy
//   shape_size*4 bytes out of the f16 (2-byte) source buffer -> 2x over-read.
//   Under ASan this aborts with "heap-buffer-overflow READ" inside Constant::evaluate.
// Post-fix behaviour: evaluate_lower() must convert (or reject) the mismatched type
//   instead of raw-copying, so no OOB read occurs.
TEST(constant_evaluate_lower, type_mismatch_no_oob_read) {
    using namespace ov;

    const Shape shape{4};
    std::vector<float16> src(shape_size(shape), float16(1.0f));
    auto c = std::make_shared<op::v0::Constant>(element::f16, shape, src.data());

    // Wider-typed output tensor of the same shape, emulating TypeRelaxed<Constant>.
    Tensor wider_out(element::f32, shape);
    TensorVector outputs{wider_out};

    // Pre-fix: ASan flags an OOB read during the memcpy in Constant::evaluate.
    // Post-fix: completes safely (value converted / mismatch handled).
    ASSERT_NO_THROW(c->evaluate_lower(outputs));
    // TODO: once the fix lands, also assert the produced values equal the
    //       f16->f32 converted source (not a raw bit reinterpretation).
}