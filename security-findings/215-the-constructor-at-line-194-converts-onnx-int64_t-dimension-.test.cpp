// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-20 in openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// The Tensor(TensorProto&,...) ctor builds ov::Shape from int64 dims with no non-negative
// validation, so a negative dim becomes SIZE_MAX and dims:[-1,-1] overflow-wraps shape_size to 1,
// bypassing the size-mismatch guard in tensor.cpp:467. After the fix the model must be rejected.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp (ov_onnx_frontend_tests, gtest).
// TODO: add a binary fixture models/negative_dim_initializer.onnx whose initializer has
//       dims:[-1,-1] (or dims:[-1]) and matching raw_data — there is no pure-API way to inject a
//       negative protobuf dim through the C++ frontend API, so a crafted .onnx is required.
#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// Expected to FAIL pre-fix (model loads / Constant built with SIZE_MAX shape, possibly ASan abort
// downstream); PASSES post-fix because the ctor throws on the negative dimension.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_negative_initializer_dimension_rejected) {
    EXPECT_THROW(convert_model("negative_dim_initializer.onnx"), ov::Exception);
}