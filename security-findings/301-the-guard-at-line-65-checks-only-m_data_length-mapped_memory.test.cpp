// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for TOCTOU/OOB at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65-71
// Pre-fix: the post-mmap guard (line 65) checks only m_data_length vs mapped size and
//   omits m_offset < mapped_size, and line 70 uses the stale stat() file_size for the
//   fallback length. If the external-data file shrinks between file_size() (line 52)
//   and load_mmap_object() (line 62), line 69 (mapped->data()+m_offset) is OOB.
// Post-fix: all bounds derived from mapped_memory->size(); load is rejected with
//   ov::Exception / onnx::error::invalid_external_data.
//
// Lives in openvino/src/frontends/onnx/tests in the style of onnx_import.in.cpp,
// target ov_onnx_frontend_tests (built with ASan).

#include "gtest/gtest.h"
#include "onnx_utils.hpp"   // FrontEndTestUtils convert_model(...)
#include "openvino/core/except.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: This race cannot be triggered deterministically from a static .onnx fixture
//       because under non-race conditions mapped_memory->size() == file_size and the
//       line 52-55 stat-based guard already rejects an out-of-range offset. Reproducing
//       requires interposing the filesystem (truncate/symlink-swap the external data
//       file between stat() and mmap()).
//
// TODO: Provide a crafted model 'external_data_offset_past_mmap.onnx' whose TensorProto
//       external_data sets offset=N (e.g. 0x1000), length=0, location pointing at a
//       helper-controlled file, AND a test harness that shrinks that file below N after
//       file_size() but before load_mmap_object() (e.g. via an LD_PRELOAD/syscall shim
//       or a FUSE/cooperative filesystem). Without that interposition this assertion
//       cannot fail pre-fix.
//
// TODO: Confirm the exact convert_model helper signature + fixture path from
//       onnx_import.in.cpp before enabling.
TEST(onnx_external_data, DISABLED_offset_past_truncated_mmap_is_rejected) {
    // Expected post-fix behaviour: even if the mapped region is smaller than the
    // stat()-reported size, an offset that lands outside the live mapping must throw.
    EXPECT_THROW(convert_model("external_data/external_data_offset_past_mmap.onnx"),
                 ov::Exception);
}
