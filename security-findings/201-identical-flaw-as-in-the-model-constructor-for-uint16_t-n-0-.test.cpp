// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for compiled_model.cpp:160-163 (CWE-835/CWE-190).
// Pre-fix: ExecutionConfig with ov::num_streams(70000) reaches the cache-restore
// (and model) ctor loop `for (uint16_t n = 0; n < get_num_streams(); n++)`, where
// the uint16_t counter wraps at 65535 -> infinite loop / unbounded m_graphs growth.
// Post-fix (int32_t counter + num_streams upper-bound validator): num_streams is
// rejected/clamped, so compile completes with a bounded stream count (or throws).
//
// NOTE: This test cannot directly assert an infinite loop (it would hang). It instead
// pins the *fix's* observable contract: an out-of-range num_streams is rejected.
// TODO: confirm exact target/include paths against intel_gpu/tests/unit/ before use.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

// TODO: if intel_gpu unit tests use a fixture (e.g. a cldnn engine fixture under
// intel_gpu/tests/unit/), inherit from it instead of constructing ov::Core directly.
static std::shared_ptr<Model> make_trivial_model() {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 8});
    auto r = std::make_shared<op::v0::Relu>(p);
    auto res = std::make_shared<op::v0::Result>(r);
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{p});
}

TEST(intel_gpu_num_streams, rejects_out_of_range_num_streams) {
    ov::Core core;
    auto model = make_trivial_model();
    // 70000 > UINT16_MAX: pre-fix this drives the wrapping loop at compiled_model.cpp:160.
    // Post-fix the value must be rejected (validator) — assert it throws rather than hangs.
    // TODO: if the fix clamps instead of throwing, replace EXPECT_THROW with a check that
    //       the compiled model's ov::num_streams property is <= the sane max (e.g. 1024).
    EXPECT_THROW(
        { auto cm = core.compile_model(model, "GPU", {ov::num_streams(70000)}); (void)cm; },
        ov::Exception);
}
