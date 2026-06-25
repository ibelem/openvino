// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 in Tensor::get_ov_constant
//   targets/openvino/src/frontends/onnx/frontend/src/core/tensor.cpp:438 (shape_size wrap)
//   targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194 (unvalidated int64 dims -> size_t)
//
// Pre-fix: dims=[2, 0x8000000000000001] wrap to shape product 2, matching 8 bytes of
// inline FLOAT raw_data, so the size check at tensor.cpp:467 does NOT throw and a
// Constant with a giant declared shape over an 8-byte buffer is created; downstream
// indexing reads OOB (ASan heap-buffer-overflow). Post-fix: the frontend must reject
// the overflowing/absurd dimension and throw ov::Exception during convert_model.
//
// This needs a crafted binary .onnx fixture, so it is a SKELETON.

#include "onnx_utils.hpp"          // FrontEndTestUtils / convert_model helpers used by onnx_import.in.cpp
#include "common_test_utils/test_assertions.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: create the fixture model under
//   src/frontends/onnx/tests/models/tensor_dims_overflow.onnx
// with an initializer: data_type=FLOAT, dims=[2, 0x8000000000000001],
// raw_data = 8 bytes (two float32 values). On 64-bit the size_t product wraps to 2.
TEST(onnx_importer, DISABLED_tensor_dims_integer_overflow_rejected) {
    // TODO: replace with the project's model-path helper (see onnx_import.in.cpp).
    EXPECT_THROW(convert_model("tensor_dims_overflow.onnx"), ov::Exception);
}
