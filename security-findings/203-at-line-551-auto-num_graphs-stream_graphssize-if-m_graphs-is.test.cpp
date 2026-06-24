// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 at
//   openvino/src/plugins/intel_gpu/src/plugin/sync_infer_request.cpp:551-555
// Encodes the fix: with NUM_STREAMS=0 the GPU CompiledModel must NOT leave
// m_graphs empty / must reject inference cleanly instead of executing
// `stream_id % 0` (SIGFPE/UB) or indexing an empty vector at line 555.
// Pre-fix: crashes (FPE / ASan container-overflow on stream_graphs[0]).
// Post-fix: the OPENVINO_ASSERT / num_streams clamp turns it into a throw.
//
// NOTE: requires a real GPU device; symbols/macros below are best-effort and
// must be confirmed against intel_gpu/tests/unit/ before use.
#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

TEST(gpu_sync_infer_request, num_streams_zero_does_not_divide_by_zero) {
    ov::Core core;
    // TODO: replace with a small valid model fixture available in the GPU unit tree
    //       (e.g. ngraph::builder helpers used elsewhere in intel_gpu/tests/unit/).
    std::shared_ptr<ov::Model> model = nullptr; // TODO: build a trivial 1-op model
    ASSERT_NE(model, nullptr) << "TODO: construct test model";

    // Explicitly request zero streams — currently survives execution_config finalize.
    ov::AnyMap cfg = { ov::num_streams(ov::streams::Num(0)) };

    // After the fix this should either clamp to >=1 (so infer succeeds) or throw
    // at compile/infer time; it must NEVER crash with FPE.
    ASSERT_ANY_THROW({
        auto compiled = core.compile_model(model, "GPU", cfg);
        auto req = compiled.create_infer_request();
        req.infer(); // setup_stream_graph(): stream_id % num_graphs with num_graphs==0
    });
}
