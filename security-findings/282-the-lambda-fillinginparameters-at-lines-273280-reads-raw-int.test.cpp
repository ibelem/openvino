// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing pad-range validation at
// openvino/src/plugins/intel_cpu/src/nodes/pad.cpp:273-280 (fillingInParameters)
// and the resulting srcODims wrap at pad.cpp:390 / OOB read at pad.cpp:522,567.
//
// Pre-fix: running a CONSTANT-mode Pad whose DYNAMIC pads_begin holds a value
// more negative than -srcDim (with pads_end compensating so the output dim stays
// positive and allocation succeeds) makes srcODims[i] = padsBegin[i] + srcDims[i]
// wrap to ~SIZE_MAX, the interior guard never fires, and exec reads past srcData
// (ASan: heap-buffer-overflow READ in cpu_memcpy inside padConstant*).
// Post-fix: paramsInitialization rejects the out-of-range pad and the infer call
// throws ov::Exception instead.
//
// TODO: confirm exact CPU unit-test target/harness and helper symbols by reading
// openvino/src/plugins/intel_cpu/tests/unit/ (target name is ov_cpu_unit_tests).
// TODO: replace the pseudo-graph construction below with the repo's actual
// node-test fixture (e.g. build an ov::op::v1::Pad with Parameter pads_begin/
// pads_end, compile on CPU, set pads_begin = {-(dim+5)}, pads_end = {+10}).

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(intel_cpu_Pad, dynamic_pads_out_of_range_negative_is_rejected) {
    // data: shape [8]
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{8});
    // dynamic pad inputs (1-D, length 1)
    auto pb = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pe = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto pad = std::make_shared<op::v1::Pad>(data, pb, pe, op::PadMode::CONSTANT);
    auto model = std::make_shared<Model>(OutputVector{pad}, ParameterVector{data, pb, pe});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    Tensor t_data(element::f32, Shape{8});
    Tensor t_pb(element::i32, Shape{1});
    Tensor t_pe(element::i32, Shape{1});
    // padsBegin = -(srcDim + 5) = -13, padsEnd = +10  -> output dim = 8-13+10 = 5 (>0, alloc ok)
    // but srcODims = padsBegin + srcDim = -13 + 8 = -5 -> wraps to ~SIZE_MAX pre-fix.
    t_pb.data<int32_t>()[0] = -13;
    t_pe.data<int32_t>()[0] = 10;
    req.set_input_tensor(0, t_data);
    req.set_input_tensor(1, t_pb);
    req.set_input_tensor(2, t_pe);

    // Post-fix: pad value out of [-srcDim, ...] range must be rejected, not OOB-read.
    ASSERT_ANY_THROW(req.infer());
}
