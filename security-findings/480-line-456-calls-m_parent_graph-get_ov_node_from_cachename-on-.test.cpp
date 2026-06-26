// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-674 uncontrolled recursion in
// openvino/src/frontends/onnx/frontend/src/core/graph.cpp:456 (and :449).
// Pre-fix: importing a model with deeply nested Loop/If subgraphs whose inner
// body references an outermost-scope tensor recurses N frames through
// Subgraph::get_ov_node_from_cache and overflows the stack (ASan: stack-overflow).
// Post-fix: a MAX_SUBGRAPH_DEPTH guard makes convert_model reject the model with
// an ov::Exception instead of crashing.
//
// NOTE: this requires a crafted binary fixture (deeply_nested_loops.onnx) that is
// not present in the tree; emitted as a SKELETON.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// TODO: generate models/deeply_nested_loops.onnx with ~1000 nested Loop bodies,
//       each innermost body referencing a tensor defined only in the top-level
//       graph (forces Subgraph::get_ov_node_from_cache to walk all N parents).
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_deeply_nested_subgraph_recursion_guard) {
    EXPECT_THROW(convert_model("deeply_nested_loops.onnx"), ov::Exception);
}
