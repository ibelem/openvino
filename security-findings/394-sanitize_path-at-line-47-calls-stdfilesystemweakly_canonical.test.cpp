// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression intent for TOCTOU at tensor_external_data.cpp:47-62 / 74-113.
// A deterministic race is not testable in gtest; instead this skeleton encodes
// the containment invariant the fd-based fix must still satisfy: an external-data
// 'location' that resolves (via a symlink in model_dir) outside the base dir must
// be REJECTED, and once the fix opens with O_NOFOLLOW / fd-based I/O the symlinked
// final component must not be followed.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted .onnx whose tensor external_data.location names
//       "weights_toctou.bin" (a file absent at parse time) under models/.
static std::string s_model = "external_data/external_data_toctou.onnx";

TEST(onnx_external_data_toctou, symlink_escapes_base_is_rejected) {
    // TODO: in the test fixture, create models/external_data/weights_toctou.bin
    //       as a symlink pointing outside the model directory (e.g. to a temp
    //       file standing in for /etc/shadow). std::filesystem::create_symlink.
    //       Skip on platforms/CI without symlink privilege.
    //
    // Pre-fix: sanitize_path passes (final component absent / appended verbatim),
    //          then file_size()+load_mmap_object() follow the symlink => leak.
    // Post-fix: open(O_NOFOLLOW)+fstat+mmap(fd) refuses the symlinked component,
    //          surfacing as ov::Exception / invalid_external_data.
    EXPECT_THROW(convert_model(s_model), ov::Exception);
}
