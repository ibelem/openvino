// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read in ov::frontend::onnx detail::__get_raw_data<T>
// (targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:72).
// The flaw: element_count is derived from get_onnx_data_size(onnx_data_type)
// but the const T* iterator advances by sizeof(T); a raw tensor declared with a
// 1-byte ONNX type (e.g. INT8/UINT8/FLOAT8E4M3FN) but consumed via get_data<int64_t>()
// reads count*8 bytes from a count-byte raw_data buffer -> heap-buffer-overflow.
//
// This encodes the fix as: loading a crafted .onnx whose initializer/sparse-indices
// tensor has has_raw_data()==true and a data_type whose byte size is smaller than the
// type the importer reads it as must be REJECTED with ov::Exception (not silently OOB-read).
//
// Pre-fix: triggers ASan heap-buffer-overflow inside __get_raw_data.
// Post-fix: convert_model throws ov::Exception due to size/type validation.
//
// TODO: provide the crafted fixture models/onnx/tensor_raw_data_type_mismatch.onnx :
//   - a Constant/initializer tensor (or SparseTensor indices) with raw_data of N bytes,
//     data_type = INT8 (or FLOAT8E4M3FN), but referenced where the frontend calls
//     get_data<int64_t>() (e.g. sparse-tensor indices in opset_13 Constant).
//   The exact onnx must be built with the onnx python helper; binary fixture cannot be
//   authored inline here.

#include "onnx_utils.hpp"  // TODO: confirm helper header used by onnx_import.in.cpp (convert_model)
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

// static std::string s_manifest = "...";  // TODO: match onnx_import.in.cpp manifest pattern

OPENVINO_TEST(onnx_import_validation, tensor_raw_data_type_size_mismatch_is_rejected) {
    // TODO: place crafted model under the frontend test models dir.
    EXPECT_THROW(convert_model("tensor_raw_data_type_mismatch.onnx"), ov::Exception);
}