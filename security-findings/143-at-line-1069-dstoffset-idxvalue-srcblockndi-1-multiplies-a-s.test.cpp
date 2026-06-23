// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for OOB write in ScatterUpdate::scatterNDUpdate (ReduceNone)
// Cited unchecked code: scatter_update.cpp:1063-1069 (no per-element range
// check on idxValue) guarded only by scatter_update.cpp:1073
// (`dstOffset < elementsCount`, start-offset only). The fix adds
// `CPU_NODE_ASSERT(idxValue >= 0 && (size_t)idxValue < srcDataDim[i], ...)`.
//
// This assertion encodes: a ScatterNDUpdate whose indices tensor carries an
// out-of-range / overflow-inducing coordinate must be REJECTED (throw), not
// silently executed. Pre-fix this triggers an ASan heap-buffer-overflow at the
// cpu_memcpy on scatter_update.cpp:1079; post-fix it throws ov::Exception.
//
// SKELETON: exact intel_cpu unit-test fixture/helpers for constructing and
// inferring a single-node ScatterNDUpdate graph must be copied from the
// surrounding tests under intel_cpu/tests/unit/ — symbols below are placeholders.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/scatter_nd_update.hpp"

using namespace ov;

TEST(intel_cpu_ScatterNDUpdate, RejectsOutOfRangeIndex_OOBWrite) {
    // data shape [4, 8] -> elementsCount = 32, srcBlockND[k=1] = 8 (>1 so a
    // spill past the end is a true OOB write, not just a wrong in-bounds write).
    // TODO: build a single ScatterNDUpdate op (reduction == NONE) via the
    //       intel_cpu test graph helper used in intel_gpu/intel_cpu unit tests.
    auto data    = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8});
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 1});
    auto updates = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto scatter = std::make_shared<op::v3::ScatterNDUpdate>(data, indices, updates);
    auto model   = std::make_shared<Model>(OutputVector{scatter},
                                           ParameterVector{data, indices, updates});

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // TODO: fill data/updates with valid buffers.
    // Craft the malicious index: a large positive int64 whose product with the
    // stride either wraps uint64 or drives dstOffset to ~elementsCount-1 so the
    // 8-element copy spills past the 32-element destination.
    Tensor idxT(element::i64, Shape{1, 1});
    idxT.data<int64_t>()[0] = static_cast<int64_t>((1LL << 33) + 1); // overflow / OOB driver
    req.set_input_tensor(1, idxT);

    // Pre-fix: ASan heap-buffer-overflow inside cpu_memcpy (scatter_update.cpp:1079).
    // Post-fix: CPU_NODE_ASSERT rejects the index -> ov::Exception.
    EXPECT_ANY_THROW(req.infer());
}
