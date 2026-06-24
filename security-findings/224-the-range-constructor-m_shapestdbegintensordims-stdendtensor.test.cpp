// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-195 at
// openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:194
// where `m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}` narrows
// negative int64_t protobuf dims to SIZE_MAX with no sign check.
//
// Pre-fix: loading a model whose initializer has dims [-1,-1] and a 1-element
//          float_data buffer builds an ov::op::v0::Constant with shape
//          {SIZE_MAX,SIZE_MAX} (size-mismatch guard at tensor.cpp:467 is bypassed
//          because shape_size wraps to 1) -> corrupted Constant / OOB downstream
//          (ASan heap-buffer-overflow on a later read of the constant).
// Post-fix: FRONT_END_GENERAL_CHECK(dim >= 0, ...) throws ov::Exception during
//           convert_model, so the model is rejected cleanly.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan), style of onnx_import.in.cpp.
//
// This needs a crafted .onnx fixture, so it is a SKELETON.

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // FrontEndTestUtils::convert_model, manifest macros

using namespace ov::frontend::onnx::tests;

// TODO: provide fixture model 'negative_dims_initializer.onnx' under
//       frontend/onnx/tests/models/ containing an initializer tensor with
//       dims = [-1, -1], data_type = FLOAT, float_data = [1.0f].
TEST(onnx_import_negative_dims, initializer_negative_dims_rejected) {
    // Pre-fix: convert_model succeeds and yields a Constant with SIZE_MAX dims;
    // a subsequent shape/data access trips ASan. Post-fix: throws ov::Exception.
    EXPECT_THROW(convert_model("negative_dims_initializer.onnx"), ov::Exception);
}