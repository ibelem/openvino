// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for CWE-125 OOB read in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:65-71
// The post-mmap guard at line 65 omits the m_offset bound, so a mapping smaller
// than the file_size validated at lines 52-54 lets SharedBuffer index past the
// mapping at line 69 (data()+m_offset) / line 70 (file_size - m_offset).
//
// SKELETON: a deterministic, self-contained gtest cannot be authored cleanly
// because triggering the flaw requires mapped_memory->size() to diverge from
// the file_size stat — i.e. either the TOCTOU file-swap race between l.52 and
// l.62, or a cache entry (l.59-60) whose backing file grew after being mmapped.
// Neither is reproducible from a single static .onnx + data file in the normal
// ov_onnx_frontend_tests flow (there file_size == mapping size, so the bug is
// masked). The items below name exactly what is missing.
//
// Suggested location: src/frontends/onnx/tests/onnx_import.in.cpp
// (use the existing convert_model() helper as in that file).

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"  // for convert_model(...) used across onnx_import.in.cpp

using namespace ov::frontend::onnx::tests;

TEST(${BACKEND_NAME}, onnx_external_data_offset_oob_after_mapping_shrinks) {
    // TODO(fixture): provide a crafted model 'external_data_offset_oob.onnx'
    //   whose tensor external_data sets a LARGE offset+length that is valid
    //   against the INITIAL data file size, e.g. offset=900, length=50,
    //   data file 'data.bin' initially 1000 bytes.
    // TODO(race/grow): after convert begins (or via a pre-cached mapping whose
    //   file later shrank to 100 bytes), the mmap must be smaller than the
    //   validated file_size. There is no clean public hook to force this in
    //   the frontend test harness; a unit test on TensorExternalData directly
    //   (constructing it with offset=900,length=50 and pointing at a 100-byte
    //   file while a stale 1000-byte mapping sits in the MappedMemoryHandles
    //   cache) is the realistic encoding. That requires friending/exposing
    //   load_external_mmap_data + a hand-built cache map.
    //
    // Expected with the fix in place: the offset re-check at the post-mapping
    // guard rejects the load -> convert_model throws.
    // Pre-fix (ASan build): heap-buffer-overflow READ in
    //   ov::SharedBuffer ctor / subsequent tensor access (data()+m_offset).
    EXPECT_THROW(convert_model("external_data_offset_oob.onnx"), ov::Exception);
}
