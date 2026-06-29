// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for tensor_external_data.cpp:65-69:
// The stale-cache path allows m_offset to be unchecked against mapped_memory->size(),
// causing an OOB read at line 69 (`mapped_memory->data() + m_offset`).
// Pre-fix: this test triggers a heap-buffer-overflow (detected by ASan) or silent OOB.
// Post-fix: load_external_mmap_data throws ov::Exception (invalid_external_data).

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

// Internal header — adjust include path to match build-tree layout
#include "utils/tensor_external_data.hpp"
#include "exceptions.hpp"           // error::invalid_external_data
#include "openvino/util/mmap_object.hpp"

namespace ov_onnx_ext_data_test {

// Write `size` bytes of value 0xAB to a temp file; return the path.
static std::filesystem::path write_tmp_file(const std::string& name, std::size_t size) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(size, static_cast<char>(0xAB));
    f.write(buf.data(), static_cast<std::streamsize>(size));
    return path;
}

} // namespace ov_onnx_ext_data_test

using namespace ov_onnx_ext_data_test;

// Validates that a stale cache entry (mapped_memory->size() < file_size)
// cannot be exploited via an unchecked m_offset.
//
// Setup:
//   1. Write a 100-byte file; mmap it; insert into cache (size = 100 B).
//   2. Grow the file to 1000 B on disk.
//   3. Construct TensorExternalData with offset=900, length=9.
//      - First guard: 900+9 <= file_size(1000) → passes.
//      - Second guard pre-fix: 9 > 100 → false → passes; line 69 OOBs by ~809 bytes.
//      - Second guard post-fix: 900 > 100-9 → true → throws.
TEST(OnnxExternalDataMmap, StaleCache_LargeOffsetWithSmallLength_ThrowsOrASan) {
    const std::string filename = "ov_ext_data_stale_cache.bin";
    auto file_path = write_tmp_file(filename, 100);
    auto model_dir  = file_path.parent_path();

    // Pre-populate cache with the 100-B mapping.
    auto small_mapping = ov::load_mmap_object(file_path);
    ASSERT_EQ(small_mapping->size(), std::size_t{100})
        << "mmap of 100-B file must report size 100";

    auto cache = std::make_shared<
        std::map<std::filesystem::path, std::shared_ptr<ov::MappedMemory>>>();
    (*cache)[file_path] = small_mapping;

    // Grow the file to 1000 B so that ov::util::file_size() now returns 1000
    // while the cache still holds the 100-B MappedMemory.
    {
        std::ofstream f(file_path, std::ios::binary | std::ios::trunc);
        std::vector<char> buf(1000, static_cast<char>(0xCD));
        f.write(buf.data(), 1000);
    }

    // offset=900, length=9: valid against file_size(1000), OOB against mapped size(100).
    ov::frontend::onnx::detail::TensorExternalData ext_data(
        filename, /*offset=*/900, /*length=*/9);

    // Post-fix: must throw; pre-fix: ASan heap-buffer-overflow on line 69.
    EXPECT_THROW(
        { ext_data.load_external_mmap_data(model_dir, cache); },
        ov::Exception);  // error::invalid_external_data derives from ov::Exception

    std::filesystem::remove(file_path);
}

// Validate the boundary case: offset exactly at the end of the mapped region.
TEST(OnnxExternalDataMmap, StaleCache_OffsetAtMappedEnd_ThrowsOrASan) {
    const std::string filename = "ov_ext_data_boundary.bin";
    auto file_path = write_tmp_file(filename, 50);
    auto model_dir  = file_path.parent_path();

    auto small_mapping = ov::load_mmap_object(file_path);
    ASSERT_EQ(small_mapping->size(), std::size_t{50});
    auto cache = std::make_shared<
        std::map<std::filesystem::path, std::shared_ptr<ov::MappedMemory>>>();
    (*cache)[file_path] = small_mapping;

    // Grow to 500 B.
    {
        std::ofstream f(file_path, std::ios::binary | std::ios::trunc);
        std::vector<char> buf(500, static_cast<char>(0xEF));
        f.write(buf.data(), 500);
    }

    // offset=50 == mapped size, length=1 → OOB by 1 byte at line 69.
    ov::frontend::onnx::detail::TensorExternalData ext_data(
        filename, /*offset=*/50, /*length=*/1);

    EXPECT_THROW(
        { ext_data.load_external_mmap_data(model_dir, cache); },
        ov::Exception);

    std::filesystem::remove(file_path);
}