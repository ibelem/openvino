// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-367 TOCTOU at
//   openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:303-328
// Pre-fix: file_size() snapshot at :303 validates offset/length, but the mmap at :323
// is never re-validated, so data()+ext_data_offset at :328 can be out of bounds when the
// external-data file is smaller than the stat-time size (truncation/symlink race).
// This test asserts that loading a model whose external_data offset+length exceeds the
// ACTUAL mapped file size is rejected (throws) instead of producing an OOB pointer.
//
// NOTE (skeleton): a pure C++ unit test cannot reproduce the live race window between
// stat and mmap. Instead it must stage a crafted model + external file where the mapped
// size is smaller than offset+length so the post-mmap check fires deterministically.
// TODO: provide binary fixtures (see TODOs) — exact bytes are not derivable by reading source.

#include "onnx_utils.hpp"            // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/file_utils.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: create a temp dir containing:
//   - external_data_truncated.onnx : a TensorProto with external_data
//       location="weights.bin", offset="65536", length="1024"
//   - weights.bin : a file SMALLER than offset+length (e.g. 16 bytes) so the mmap is
//                   smaller than the protobuf-claimed range. (To exercise the static
//                   check instead, this also covers offset beyond file end.)
// The pre-fix code dereferences mapped_memory->data()+65536 -> ASan/SIGSEGV/SIGBUS.
// The post-fix code must throw before constructing m_tensor_data.
TEST(onnx_external_data, mmap_offset_beyond_actual_file_size_is_rejected) {
    const std::string model_path =
        ov::test::utils::getModelFromTestModelZoo(
            std::string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_truncated.onnx");
    // Expect a throw rather than an OOB read when offset/length exceed the mapped size.
    EXPECT_THROW(convert_model(model_path), ov::Exception);
}
