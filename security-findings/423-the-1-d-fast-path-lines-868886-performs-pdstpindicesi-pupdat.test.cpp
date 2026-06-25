// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for scatter_update.cpp:883 (1-D fast-path ScatterUpdate OOB write).
// Pre-fix: an INDICES value outside [0, srcLength) drives pdst[pindices[i]] out of
// bounds -> ASan heap-buffer-overflow. Post-fix: the node must reject the model
// (CPU_NODE_ASSERT/THROW) so the op build/infer throws ov::Exception.
//
// HARNESS: intel_cpu unit tests (target ov_cpu_unit_tests). Place under
//   openvino/src/plugins/intel_cpu/tests/unit/  next to existing single-layer/node tests.
// TODO: confirm exact fixture/helper names by reading the surrounding tests/unit tree;
//       the symbols below are illustrative and must be matched to the repo's harness.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

// Builds a model whose ScatterUpdate hits the 1-D fast path: i32 data of rank 1,
// size <= 64, i32 indices, with an out-of-range index value.
TEST(scatter_update_cpu, fast_path_oob_index_is_rejected) {
    auto data    = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto indices = op::v0::Constant::create(element::i32, Shape{1}, {-65536}); // OOB
    auto updates = op::v0::Constant::create(element::i32, Shape{1}, {0xdeadbeef});
    auto axis    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto su = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
    auto model = std::make_shared<Model>(OutputVector{su}, ParameterVector{data});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    int32_t in = 0;
    req.set_input_tensor(Tensor(element::i32, Shape{1}, &in));
    // TODO: depending on where the check lands (compile vs infer), the throw may
    // occur at compile_model or at infer(); assert on the inference call here.
    EXPECT_THROW(req.infer(), ov::Exception);
}