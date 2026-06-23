// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-191 unsigned underflow in addHiddenDims()
// at openvino/src/plugins/intel_cpu/src/nodes/strided_slice.cpp:219-220.
//
// Pre-fix: building a v1::StridedSlice with ellipsis_mask whose set bit index
// (ellipsisPos1) is >= begin.size() makes
//     afterDims  = begin.size() - ellipsisPos1 - 1   // size_t underflow -> ~2^64
//     ellipsisPos2 = inputRank - afterDims - 1        // further underflow
// and the loops at lines 226/228/231/232 read out of bounds / push_back
// unbounded, detected by ASan as a heap-buffer-overflow or OOM.
// Post-fix: the added bounds check (ellipsisPos1 >= begin.size() ->
// OPENVINO_THROW / NODE_VALIDATION_CHECK) makes graph compile reject the model.
//
// Harness: ov_cpu_unit_tests (gtest + ASan), intel_cpu/tests/unit/.
//
// SKELETON — addHiddenDims is a file-static function and cannot be called
// directly; it is only reachable through StridedSlice::initSupportedPrimitiveDescriptors().
// The exact node-construction / Graph-compile helper symbols must be copied from
// an existing intel_cpu/tests/unit node test before this will compile.

#include <gtest/gtest.h>
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(StridedSliceCpuNode, EllipsisPosBeyondBeginSizeIsRejected) {
    // input data rank 4 (>3, required to enter the line-215 branch)
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});
    // begin/end/stride constants of length 1 -> begin.size() == 1
    auto begin  = op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto end    = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto stride = op::v0::Constant::create(element::i64, Shape{1}, {1});

    // ellipsis_mask length 4 with the set bit at index 3 -> ellipsisPos1 == 3,
    // which is >= begin.size() (==1): triggers the size_t underflow at line 219.
    const std::vector<int64_t> begin_mask{0, 0, 0, 0};
    const std::vector<int64_t> end_mask{0, 0, 0, 0};
    const std::vector<int64_t> new_axis_mask{0, 0, 0, 0};
    const std::vector<int64_t> shrink_axis_mask{0, 0, 0, 0};
    const std::vector<int64_t> ellipsis_mask{0, 0, 0, 1};

    auto ss = std::make_shared<op::v1::StridedSlice>(data, begin, end, stride,
                                                     begin_mask, end_mask,
                                                     new_axis_mask, shrink_axis_mask,
                                                     ellipsis_mask);
    auto model = std::make_shared<Model>(OutputVector{ss->output(0)}, ParameterVector{data});

    // TODO: replace with the intel_cpu unit-test graph-compile helper that drives
    //       StridedSlice::initSupportedPrimitiveDescriptors() (the entry that calls
    //       addHiddenDims at strided_slice.cpp:306). Copy the exact helper /
    //       fixture symbol from an existing test under intel_cpu/tests/unit/ (e.g.
    //       a Graph/Node builder). Pre-fix this aborts under ASan; post-fix it must
    //       throw ov::Exception.
    // EXPECT_THROW(compile_cpu_node_supported_descriptors(model), ov::Exception);
    GTEST_SKIP() << "TODO: wire to intel_cpu node compile helper to reach addHiddenDims";
}
