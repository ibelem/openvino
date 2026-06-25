// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for graph_iterator_proto.cpp:304-314: a negative file_size
// (ov::util::file_size returns -1, file_util.hpp:154) combined with an absent
// 'length' (ext_data_length==0) must be REJECTED, not silently turned into
// resolved_data_length = SIZE_MAX -> allocate_data(SIZE_MAX) at line 353.
// Pre-fix: convert_model triggers std::bad_array_new_length / OOB and may not
// throw ov::Exception cleanly. Post-fix: a guarded runtime_error is raised.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
// gtest filter: --gtest_filter=*external_data_negative_filesize*

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // get_model_path / convert_model helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_import_external_data, external_data_negative_filesize_zero_length) {
    // TODO: provide a crafted fixture under models/external_data/ whose tensor's
    // external_data sets location to a path for which std::filesystem::file_size
    // returns an error (ec != 0) while the path is still openable as a stream —
    // e.g. a non-regular/special file or a fixture installed at test time as a
    // FIFO/symlink. A plain missing file is NOT sufficient because the ifstream
    // fail-check at line 347 (or load_mmap_object) would throw for a different
    // reason. The 'length' key must be ABSENT so ext_data_length defaults to 0.
    //
    // The assertion the regression encodes: the model load must fail with a
    // controlled ov::Exception/runtime_error, and must NOT reach
    // allocate_data(SIZE_MAX) (no std::bad_array_new_length / ASan abort).
    OV_EXPECT_THROW(convert_model("external_data/external_data_negative_filesize.onnx"),
                    ov::Exception,
                    testing::HasSubstr("external"));
}
