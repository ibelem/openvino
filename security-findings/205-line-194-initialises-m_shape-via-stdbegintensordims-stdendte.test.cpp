// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes fix for CWE-194/CWE-190 at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// where ov::Shape is constructed from int64_t TensorProto.dims() with no
// non-negative check, so a dim of -1 wraps to SIZE_MAX.
//
// Pre-fix: convert_model on a model whose initializer has dims:[-1] either
//   builds a Constant with a SIZE_MAX dimension (huge shape_size -> bad_alloc/
//   abort under ASan) or silently produces a corrupt shape.
// Post-fix: the frontend throws ov::Exception (FRONT_END_GENERAL_CHECK)
//   rejecting the negative dim, which this test asserts.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// NOTE: requires a crafted binary fixture (negative_initializer_dim.onnx /
//       .prototxt) with an initializer tensor declaring dims:[-1]. The .onnx
//       must be added under the frontend test models dir and referenced via
//       the test's util::path helper — see TODO below.

#include "onnx_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: create models/negative_initializer_dim.prototxt containing an
//       initializer (e.g. a FLOAT tensor named "x") whose dims field is [-1].
//       Use convert_partially/convert_model exactly as other onnx_import tests do.
TEST(onnx_import_negative_dim, initializer_negative_dim_is_rejected) {
    EXPECT_THROW(convert_model("negative_initializer_dim.onnx"), ov::Exception);
}
