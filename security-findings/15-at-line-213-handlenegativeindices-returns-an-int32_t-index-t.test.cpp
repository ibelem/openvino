// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in GatherND CPU executor.
// Encodes the fix at gather_nd.cpp:272-278 (HandleNegativeIndices) / call sites 214,257.
// Pre-fix: an indices element >= the corresponding data dim makes
//   dataIdx = srcShifts[i]*index point far past the source buffer, and
//   cpu_memcpy at gather_nd.cpp:216 reads OOB (ASan: heap-buffer-overflow READ).
// Post-fix: HandleNegativeIndices throws ov::Exception, surfaced by infer().
//
// Harness: ov_cpu_unit_tests (gtest). Build a v8 GatherND model, set a crafted
// out-of-range indices tensor, run on CPU, and assert the OOB index is rejected.
//
// TODO(symbols): confirm includes/helpers against an existing model-driven test in
//   openvino/src/plugins/intel_cpu/tests/unit/ ; the executor is a private nested
//   struct (gather_nd.h:46) so it cannot be unit-tested directly — drive via ov::Core.
// TODO(precision): data is i32 (4 bytes); shape [4,4] -> stride_0=4 elements.
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(GatherND_CPU_OOB, RejectsOutOfRangeIndex) {
    // data: [4,4] i32 ; indices: [1,2] i32 (sliceRank=2, batchDims=0)
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4, 4});
    auto idx  = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 2});
    auto gnd  = std::make_shared<op::v8::GatherND>(data, idx, /*batch_dims=*/0);
    auto res  = std::make_shared<op::v0::Result>(gnd);
    auto model = std::make_shared<Model>(ResultVector{res}, ParameterVector{data, idx});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    std::vector<int32_t> data_vals(16);
    for (int i = 0; i < 16; ++i) data_vals[i] = i;
    Tensor data_t(element::i32, Shape{4, 4}, data_vals.data());

    // Crafted: row index 1000 is far outside data dim 0 (==4).
    std::vector<int32_t> idx_vals = {1000, 0};
    Tensor idx_t(element::i32, Shape{1, 2}, idx_vals.data());

    req.set_input_tensor(0, data_t);
    req.set_input_tensor(1, idx_t);

    // Pre-fix: ASan heap-buffer-overflow READ inside cpu_memcpy (gather_nd.cpp:216).
    // Post-fix: HandleNegativeIndices throws -> infer() reports an ov::Exception.
    ASSERT_ANY_THROW(req.infer());
}
