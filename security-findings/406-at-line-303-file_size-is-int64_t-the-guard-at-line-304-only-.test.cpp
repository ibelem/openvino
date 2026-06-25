// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for graph_iterator_proto.cpp:304-314 (CWE-195/CWE-20):
// an ONNX external_data tensor whose backing file yields file_size()<=0 while
// the protobuf omits the "length" key (ext_data_length==0) must be REJECTED
// before resolved_data_length underflows to ~SIZE_MAX.
//
// Pre-fix: guard at line 304 is bypassed (file_size<=0 only checked when
//   ext_data_length>0), resolved_data_length = (size_t)file_size - offset
//   = SIZE_MAX-offset -> allocate_data(SIZE_MAX) / OOB tensor size.
// Post-fix: unconditional file_size<=0 check throws here.
//
// SKELETON: triggering file_size()==-1 needs a backing path that errors on
// std::filesystem::file_size but still satisfies the chosen memory mode, and a
// crafted .onnx that points external_data->location at it with offset set and
// length omitted. Both require fixtures this skeleton cannot embed.

#include <gtest/gtest.h>
#include "openvino/frontend/manager.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::frontend;

TEST(onnx_external_data, external_data_negative_filesize_length_omitted_is_rejected) {
    FrontEndManager fem;
    auto fe = fem.load_by_framework("onnx");
    ASSERT_NE(fe, nullptr);

    // TODO: provide a model whose external_data entry has:
    //   key "location" -> a path for which ov::util::file_size() returns -1
    //                     (e.g. a directory, or a special/unstattable file)
    //   key "offset"   -> some non-zero value
    //   (NO "length" key, so ext_data_length defaults to 0)
    // const std::string model_path = "<crafted_external_data_neg_size>.onnx";
    // auto input_model = fe->load(model_path);
    // ASSERT_NE(input_model, nullptr);

    // The convert MUST throw rather than computing resolved_data_length=SIZE_MAX.
    // EXPECT_THROW(fe->convert(input_model), ov::Exception);

    GTEST_SKIP() << "TODO: supply crafted .onnx fixture + a path that makes "
                    "ov::util::file_size() return -1 while ext_data_length==0";
}
