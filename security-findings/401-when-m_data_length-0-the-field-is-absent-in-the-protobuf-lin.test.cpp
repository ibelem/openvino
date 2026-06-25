// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes a regression for tensor_external_data.cpp:70 (load_external_mmap_data):
//   when the ONNX external_data 'length' field is absent (m_data_length==0),
//   the read length must be derived from the ACTUAL mapped size, not the stale
//   pre-mmap stat (file_size). Pre-fix, a model whose external-data file is
//   smaller than the stat-time size (TOCTOU shrink) yields a SharedBuffer that
//   overruns the mapping -> OOB read (ASan heap/anon-mapping over-read).
//
// NOTE: This defect is fundamentally a TOCTOU race (file shrunk between
//   ov::util::file_size() at line 52 and ov::load_mmap_object() at line 62),
//   which cannot be triggered deterministically from a static gtest without
//   injecting a file-shrink between those two calls. Therefore this is a
//   SKELETON: it shows the convert_model entry style and the EXPECT_THROW the
//   fix should produce once length is validated against the mapped size.

#include "onnx_utils.hpp"            // TODO: confirm helper header used by onnx_import.in.cpp
#include "common_test_utils/test_constants.hpp"
#include "gtest/gtest.h"

using namespace ov::frontend::onnx::tests;

TEST(onnx_external_data, mmap_length_absent_must_clamp_to_mapped_size) {
    // TODO: provide a crafted .onnx + companion external-data file where:
    //   - external_data has only {location, offset=0}, NO 'length' (m_data_length==0)
    //   - the external-data file is shorter than the size implied at stat time
    // Pre-fix: load_external_mmap_data builds a SharedBuffer of (file_size - m_offset)
    //   bytes over a smaller mapping -> ASan over-read on first tensor access.
    // Post-fix: length is computed from mapped_memory->size() - m_offset and the
    //   offset is bounds-checked, so the malformed/short file is rejected.
    //
    // EXPECT_THROW(convert_model("external_data/external_data_length_absent_short.onnx"),
    //              ov::Exception);
    GTEST_SKIP() << "TODO: needs crafted .onnx + short external-data fixture and a "
                    "TOCTOU-shrink injection between file_size() and load_mmap_object().";
}