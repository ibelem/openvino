// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-197/CWE-190 at gather.h:92 + gather.cpp:164,173.
// Pre-fix: a Gather data dim in [2^31, 2^32) truncates to a negative `int axisDim`,
//          and `axisDim * afterAxisSizeInBytes` (int->uint64_t) wraps to a tiny
//          stride that is handed to the JIT kernel at gather.cpp:493-494 -> OOB.
// Post-fix: axisDim is int64_t/size_t and the multiply is overflow-checked, so the
//          node construction/compilation rejects (throws) the oversized dimension.
//
// HARNESS: ov_cpu_unit_tests (gtest). TODO: confirm exact target name and the
// graph-builder helpers from src/plugins/intel_cpu/tests/unit/.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

// TODO: This test allocates an enormous data tensor to reach the >INT_MAX axis
// dimension; on memory-constrained CI it must be gated/skipped. Prefer testing
// the arithmetic guard at a unit level if the helper is exposed.
TEST(GatherNodeOverflow, AxisDimExceedingIntMaxIsRejected) {
    // axis dimension 0x80000000 (2^31) -> truncates to negative int pre-fix.
    const size_t kHugeAxisDim = static_cast<size_t>(1) << 31;
    auto data = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{ static_cast<int64_t>(kHugeAxisDim), 3 });
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{ 1 });
    auto axis = op::v0::Constant::create(element::i32, Shape{}, { 0 });
    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    auto model = std::make_shared<Model>(OutputVector{ gather }, ParameterVector{ data, indices });

    ov::Core core;
    // Pre-fix: compile/exec builds corrupted stride params -> ASan OOB / UB.
    // Post-fix: the dimension-magnitude / multiply-overflow guard throws.
    EXPECT_ANY_THROW({ auto compiled = core.compile_model(model, "CPU"); });
}
