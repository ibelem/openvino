// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the off-by-one at
//   openvino/src/plugins/intel_gpu/src/plugin/compiled_model.cpp:245-246
// Pre-fix: with ov::num_streams(0) the CompiledModel ctor leaves m_graphs empty,
//   and get_graph(0) does `m_graphs[0]` on an empty vector (OOB read; ASan: heap/container-overflow).
// Post-fix (`m_graphs.size() > n`): get_graph(0) throws ov::Exception ("Invalid graph idx: 0").
// Target: ov_gpu_unit_tests (gtest + ASan).
//
// NOTE (skeleton): a real CompiledModel needs a live GPU engine/RemoteContextImpl and a
// finalized ExecutionConfig; building one inline is non-trivial in the unit harness.
// Fill the TODOs using the existing fixtures under intel_gpu/tests/unit/.
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

using namespace ov::intel_gpu;

TEST(compiled_model_get_graph, num_streams_zero_must_not_oob) {
    // TODO: obtain a GPU engine + RemoteContextImpl::Ptr from the unit fixtures
    //       (see tests that already build a cldnn::engine / RemoteContextImpl).
    // auto engine = ...; auto ctx = std::make_shared<RemoteContextImpl>(...);

    ExecutionConfig cfg;
    cfg.set_property(ov::num_streams(0));   // unvalidated -> get_num_streams()==0
    // TODO: cfg.finalize(*engine);          // must keep num_streams==0 (no AUTO normalization)

    // TODO: build a trivial ov::Model (e.g. Parameter->Result) and the plugin ptr.
    // auto model = ...; auto plugin = ...;
    // CompiledModel cm(model, plugin, ctx, cfg);   // ctor loop pushes nothing -> m_graphs empty

    // The contract: get_graph(0) / get_runtime_model() on an empty m_graphs MUST throw,
    // not read out of bounds. Pre-fix this line is an OOB read (ASan abort); post-fix it throws.
    // EXPECT_THROW(cm.get_runtime_model(), ov::Exception);
    GTEST_SKIP() << "TODO: wire GPU engine/context/model fixtures, then remove skip.";
}
