// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes the fix for tensor_external_data.cpp:65 (CWE-367/CWE-125).
// Pre-fix: load_external_mmap_data() only checks `m_data_length > mapped_memory->size()`,
//   so a mapping whose actual size is smaller than m_offset (file shrank between the
//   file_size() stat at l.52 and load_mmap_object() at l.62) still passes line 65 and
//   line 69 computes `mapped_memory->data() + m_offset` past the mapping end -> OOB/SIGBUS.
// Post-fix: the combined guard rejects m_offset > mapped_size, so load throws ov::Exception.
//
// NOTE: this defect is race/state dependent — a fully self-contained deterministic test
// requires simulating the truncation between stat and mmap (or seeding the mmap cache with
// a smaller mapping than the on-disk file). The harness hooks below are TODOs.
//
// Harness: ov_onnx_frontend_tests, in the style of onnx_import.in.cpp.
#include "gtest/gtest.h"
// TODO: include the real frontend test fixture headers used by onnx_import.in.cpp
//       (e.g. "common_test_utils/test_control.hpp", "onnx_utils.hpp").

using namespace ov::frontend;

TEST(onnx_external_data, mmap_offset_past_mapped_size_is_rejected) {
    // TODO: build an .onnx whose TensorProto external_data has
    //       offset=0x100000 (>> file), length=16, location="data.bin".
    // TODO: place a 'data.bin' that is >= 0x100010 bytes for the pre-stat check (l.53-54)
    //       to pass, then arrange for the mmap to observe a truncated (e.g. 64-byte) file
    //       — e.g. by pre-populating the MappedMemoryHandles cache (l.57-63) with a 64-byte
    //       mapping of the same path so file_size() and mapped_memory->size() diverge.
    // TODO: invoke the mmap-enabled load path (FrontEnd::load with mmap=true ->
    //       TensorExternalData::load_external_mmap_data).
    //
    // Expected after fix: throws because m_offset (0x100000) > mapped_memory->size() (64).
    EXPECT_THROW(
        { /* TODO: convert_model("crafted_external_offset.onnx") with mmap weights */ },
        ov::Exception);
    // Pre-fix this returns a SharedBuffer over an out-of-bounds pointer; first deref -> ASan/SIGBUS.
}
