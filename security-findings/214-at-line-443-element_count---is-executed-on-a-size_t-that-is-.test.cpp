// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-191 underflow at
//   openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:443 (element_count--)
// and the unchecked negative-dim store at
//   openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194 (m_shape{begin(dims),end(dims)}).
// Pre-fix: a UINT4 initializer with dims=[-1] and no raw/inline data makes
//   element_count underflow to SIZE_MAX == shape_size({SIZE_MAX}), bypassing the
//   size-mismatch throw at tensor.cpp:467, then constructs a Constant with
//   shape {SIZE_MAX} -> std::bad_alloc / OOB (ASan).
// Post-fix: get_ov_constant() (or the Tensor ctor) must reject the input and throw ov::Exception.
//
// HARNESS: ov_onnx_frontend_tests, style of onnx_import.in.cpp (uses convert_model()).
// This requires a crafted binary fixture, so it is a SKELETON.

#include "gtest/gtest.h"
#include "onnx_utils.hpp"          // TODO: confirm helper header that defines convert_model()

using namespace ov::frontend::onnx::tests;  // TODO: confirm namespace of convert_model

// TODO: create models/crafted_int4_neg_dim.onnx (read-only fixture):
//   - one initializer, data_type = UINT4 (21) [or INT4 (22)]
//   - dims: [-1]   (single int64 dimension = -1)
//   - NO raw_data, NO int32_data  (so get_data_size()==0)
//   - reference it as a graph output/Identity so get_ov_constant() runs.
TEST(onnx_import_crafted, int4_negative_dim_underflow_is_rejected) {
    // Pre-fix this either underflows element_count to SIZE_MAX and constructs a
    // Constant{shape=SIZE_MAX} (ASan: allocation-size-too-big / SEGV), or silently
    // mis-sizes. Post-fix it must throw a frontend error instead.
    EXPECT_THROW(convert_model("crafted_int4_neg_dim.onnx"), ov::Exception);
}
