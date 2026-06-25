// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the missing sign/range check at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// (Tensor::Tensor: int64 dims copied into ov::Shape<size_t> with no >=0 check).
//
// Pre-fix: a TensorProto initializer with dims=[-1,-1] sign-extends to
// m_shape={SIZE_MAX,SIZE_MAX}; shape_size wraps to 1 and the model loads,
// producing a Constant with a corrupt shape (no exception thrown).
// Post-fix: the frontend must reject the negative dim and throw ov::Exception.
//
// This is a SKELETON: it requires a crafted ONNX model containing an initializer
// whose TensorProto.dims = [-1,-1] with a single matching data element. That
// binary fixture cannot be authored inline here, hence the TODO below.
//
// Style follows onnx_import.in.cpp (ov_onnx_frontend_tests).
#include "common_test_utils/test_control.hpp"
#include "onnx_utils.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: add models/negative_initializer_dim.onnx to the onnx test models dir.
//       It must contain one initializer with TensorProto.dims = [-1, -1]
//       (int64 0xFFFFFFFFFFFFFFFF each) and a single float element of data,
//       so that the pre-fix shape_size wraparound (==1) passes tensor.cpp:467.
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_initializer_negative_dim_is_rejected) {
    EXPECT_THROW(convert_model("negative_initializer_dim.onnx"), ov::Exception);
}
