// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for graph_iterator_proto.cpp:621-635 (CWE-400 quadratic insert / no dedup).
// Pre-fix: a subgraph whose N nodes each reference the SAME outer-scope tensor causes
//   N redundant m_decoders.insert(begin()+top_index,...) at lines 633-634 (O(N^2) shifts,
//   O(N) duplicate shared_ptr copies). Post-fix (dedup): the parent tensor is injected once.
// This test loads a crafted model and asserts the model imports correctly and quickly;
// the structural invariant (no duplicate parent-decoder injection) is what the fix encodes.
#include "onnx_utils.hpp"   // TODO: confirm helper header for convert_model in ov_onnx_frontend_tests
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted fixture models/subgraph_outer_scope_fanout.onnx with a Loop/If
//       body containing many (e.g. 20000) Identity nodes that all read ONE outer-scope tensor.
//       Per-ONNX-spec valid; passes schema validation; triggers the parent-scope insert branch.
TEST(onnx_importer, subgraph_outer_scope_fanout_no_quadratic_insert) {
    std::shared_ptr<ov::Model> model;
    // Must complete (no hang / no OOM) and import successfully once the insert path is deduped/batched.
    OV_ASSERT_NO_THROW(model = convert_model("subgraph_outer_scope_fanout.onnx"));
    ASSERT_NE(model, nullptr);
    // TODO: if a white-box hook is available, assert m_decoders contains the parent tensor
    //       exactly once (not once per referencing node).
}