// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 divide-by-zero in
//   openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129
//   iter_count = full_dims[axis] / abs_stride;  // abs_stride==0 -> int div-by-zero
// stride is taken unchecked from SliceInputDescription::m_stride (line 549) /
// ConcatOutputDescription::m_stride (line 521). Core validate_and_infer_types
// never guards stride!=0 (core/src/op/tensor_iterator.cpp:98 divides by part_size).
//
// This test builds a TensorIterator whose body slices input axis with stride=0
// and runs it on the CPU plugin. Pre-fix: SIGFPE / UBSan integer-divide-by-zero
// during compile_model->createPrimitive->prepareParams->prepareInputPorts.
// Post-fix: the node validation rejects stride==0 with an ov::Exception.
//
// Target test harness: ov_cpu_unit_tests (intel_cpu/tests/unit).
// NOTE: SKELETON — exact builder helpers / include paths must be confirmed by
// reading intel_cpu/tests/unit and core SubGraphOp builder usage before use.

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(TensorIteratorCpu, SliceInputStrideZeroIsRejected) {
    // ---- body: single param -> relu -> result (shape [1, 4]) ----
    auto body_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 4});
    auto body_relu = std::make_shared<op::v0::Relu>(body_param);
    auto body_res = std::make_shared<op::v0::Result>(body_relu);
    auto body = std::make_shared<Model>(ResultVector{body_res}, ParameterVector{body_param});

    // ---- outer TI input [3, 4] sliced on axis 0 ----
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    auto ti = std::make_shared<op::v0::TensorIterator>();
    ti->set_body(body);

    // CRAFTED: stride = 0 (start=0, stride=0, part_size=1, end=-1, axis=0)
    // SliceInputDescription(input_idx, body_param_idx, start, stride, part_size, end, axis)
    ti->set_sliced_input(body_param, data, /*start=*/0, /*stride=*/0,
                         /*part_size=*/1, /*end=*/-1, /*axis=*/0);
    auto out = ti->get_iter_value(body_res, -1);

    auto model = std::make_shared<Model>(OutputVector{out}, ParameterVector{data});

    Core core;
    // Pre-fix this compile (which triggers createPrimitive/prepareParams) crashes
    // via integer divide-by-zero at tensoriterator.cpp:129. Post-fix it must throw.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
