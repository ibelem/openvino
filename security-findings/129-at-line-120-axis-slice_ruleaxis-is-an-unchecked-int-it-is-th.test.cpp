// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-129 at openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129
// (PortIteratorHelper uses PortMap.axis as an unchecked index into full_dims / dnnl dims[]).
// Pre-fix: a TensorIterator with a SliceInputDescription whose m_axis is negative-in-range
// (e.g. -2 on a rank-4 input) loads fine in core (PartialShape::operator[] normalizes it) but
// the CPU plugin uses the raw -2 as a std::vector index -> size_t(-2) OOB (ASan heap-buffer-overflow).
// Post-fix: createPrimitive normalizes/validates the axis and the graph either runs correctly or
// throws ov::Exception instead of corrupting memory.
//
// TODO: This is a SKELETON. Building a TensorIterator body graph + SliceInputDescription with a
//       crafted m_axis requires the intel_cpu unit harness (ov_cpu_unit_tests). Fill in:
//       - construction of the inner body ov::Model (Parameter -> some op -> Result),
//       - ov::op::v0::TensorIterator with set_sliced_input(param, data, start, stride, part, end, AXIS),
//         where AXIS is the malicious value (-2) [confirm exact set_sliced_input signature in
//         openvino/src/core/include/openvino/op/tensor_iterator.hpp],
//       - compile_model on CPU and infer to force createPrimitive()/PortIteratorHelper.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(TensorIteratorAxisBounds, NegativeInRangeSliceAxisIsRejectedOrNormalized) {
    using namespace ov;
    // TODO: build inner body model
    // auto body_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1,1,1,1});
    // auto body_res   = std::make_shared<op::v0::Result>(body_param);
    // auto body = std::make_shared<Model>(ResultVector{body_res}, ParameterVector{body_param});

    // auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4,1,1,1});
    // auto ti = std::make_shared<op::v0::TensorIterator>();
    // ti->set_body(body);
    // const int64_t MALICIOUS_AXIS = -2; // legal-after-normalization for core, raw OOB in CPU plugin
    // ti->set_sliced_input(body_param, data, 0, 1, 1, -1, MALICIOUS_AXIS);
    // ti->get_concatenated_slices(body_res, 0, 1, 1, -1, MALICIOUS_AXIS);
    // auto model = std::make_shared<Model>(ti->outputs(), ParameterVector{data});

    // Core core;
    // auto compiled = core.compile_model(model, "CPU");
    // auto req = compiled.create_infer_request();
    // req.set_input_tensor(Tensor(element::f32, Shape{4,1,1,1}));
    // Pre-fix: OOB inside PortIteratorHelper (ASan). Post-fix: clean run or controlled throw.
    // EXPECT_NO_FATAL_FAILURE(req.infer());
    GTEST_SKIP() << "TODO: complete TensorIterator construction per intel_cpu unit harness";
}
