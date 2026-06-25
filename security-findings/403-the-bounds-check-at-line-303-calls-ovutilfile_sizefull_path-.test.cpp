// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression target: openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:357-360
// extract_tensor_external_data() reads external tensor data into an uninitialized
// `new uint8_t[size]` buffer without verifying seekg()/read() success or gcount().
// The fix adds: throw if seekg fails, and throw if read fails / gcount != expected size.
//
// What this test encodes:
//   Loading a model whose external_data file does NOT deliver the declared number
//   of bytes at read() time must be REJECTED (throw ov::Exception / runtime_error),
//   instead of silently returning a buffer with uninitialized heap residue.
//
// NOTE: This cannot be reproduced with an ordinary static fixture, because the
// bounds check at graph_iterator_proto.cpp:304-305 validates the declared
// offset/length against ov::util::file_size() BEFORE the read and throws on any
// mismatch. A genuine short read only occurs when the file is truncated between
// file_size() (cpp:303) and read() (cpp:359) — a TOCTOU race — or via a special
// file (FIFO/device) whose seekg/read short-transfers. Both require fixture
// machinery not expressible as a plain .onnx, so this is a SKELETON.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: place alongside the other external-data tests in onnx_import.in.cpp
// and register MANIFEST entry; target binary: ov_onnx_frontend_tests.
OPENVINO_TEST(${FRONTEND_NAME}_onnx_import, model_external_data_short_read_is_rejected) {
    // TODO: build a crafted model + external data file such that the external
    //       data stream delivers FEWER bytes than declared at read() time while
    //       still passing the file_size() bounds check at cpp:304-305.
    //       Options:
    //         (a) a FIFO/named pipe or /dev-style special file as external_data location
    //             whose file_size() reports the declared length but seekg/read short-transfer;
    //         (b) a test harness hook that truncates the backing file between
    //             ov::util::file_size() and ifstream::read() (TOCTOU).
    //       Neither is a static .onnx fixture, hence the TODO.
    //
    // Pre-fix expectation: convert_model() returns successfully and the weights
    //   tensor contains uninitialized heap bytes (catchable under MSan/ASan as a
    //   use-of-uninitialized-value when the constant is consumed).
    // Post-fix expectation: convert_model() throws because read() short-transfers
    //   (gcount() != declared size) or seekg() fails.
    EXPECT_THROW(convert_model("external_data/short_read_external_data.onnx"),
                 ov::Exception);
}
