// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB read in Gather::exec1DCase (gather.cpp:987).
// Pre-fix: with a v7::Gather (reverseIndexing==false), 1-D int32 data of size 4
// and a 1-D int32 indices tensor containing a negative value (-1), exec1DCase
// sets ii=axisDim=4 (line 984) and reads psrc[4] — 4 bytes past the buffer end.
// ASan reports a heap-buffer-overflow READ. Once the fix adds the
// `if ((size_t)ii >= axisDim) { pdst[i]=0; continue; }` guard, the read is
// suppressed and the out-of-range lane is zeroed.
//
// TODO: confirm exact target/harness by reading
//       openvino/src/plugins/intel_cpu/tests/unit/ (target ov_cpu_unit_tests)
//       and the single-layer-test helpers used for Gather there.
// TODO: the 1-D int32 fast path (canOptimize1DCase) requires dataSrcRank<=1,
//       i32 precision, and dims<=64 — build the ov::Model accordingly.

#include <gtest/gtest.h>
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"
// TODO: include the intel_cpu unit-test fixture headers that let you compile
//       and infer a model on the CPU plugin (see tests/unit/ examples).

TEST(GatherCpu1DCase, NegativeIndexNoOobRead) {
    using namespace ov;
    // 1-D i32 data, size 4  -> axisDim = 4, qualifies for canOptimize1DCase
    auto data = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    // 1-D i32 indices, single value -1 (out-of-range / sentinel trigger)
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-1});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto gather = std::make_shared<op::v7::Gather>(data, indices, axis); // v7 => reverseIndexing=false
    auto model = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});

    // TODO: compile `model` on the CPU plugin and run inference with a 4-element
    //       int32 input. Pre-fix this triggers an ASan heap-buffer-overflow READ
    //       inside Gather::exec1DCase at gather.cpp:987. Post-fix the output lane
    //       for the -1 index must be 0 and no OOB access occurs.
    //   ov::Core core; auto cm = core.compile_model(model, "CPU");
    //   auto req = cm.create_infer_request();
    //   ... set input {10,20,30,40}; req.infer();
    //   EXPECT_EQ(req.get_output_tensor(0).data<int32_t>()[0], 0);
    SUCCEED() << "Skeleton — wire up CPU compile/infer per ov_cpu_unit_tests harness.";
}
