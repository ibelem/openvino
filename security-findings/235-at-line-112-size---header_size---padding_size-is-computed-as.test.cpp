// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-681 at single_file_storage.cpp:112 / :118 (and :191).
// Pre-fix: a crafted cache file whose Blob TLV length field > INT64_MAX makes
//   static_cast<std::streamoff>(size - header - pad) negative (line 112), which
//   is then stored as ~UINT64_MAX via static_cast<uint64_t> at line 118, and
//   propagated to read_cache_entry:259 as an invalid shape dimension.
// Post-fix: build_content_index must reject the oversized residual, so loading
//   the cache must NOT yield a usable blob entry / must throw on corrupt input.
//
// TODO: confirm the ov_*_tests target and existing test file for src/inference
//       (e.g. the inference unit-test tree) and the SingleFileStorage include path.
// TODO: SingleFileStorage::m_blob_index/build_content_index are private; this test
//       drives the public surface (ctor -> initialize() -> read_cache_entry()).
#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "openvino/runtime/single_file_storage.hpp"  // dev_api

using ov::runtime::SingleFileStorage;

namespace {
template <typename T>
void put(std::vector<char>& b, T v) {
    const char* p = reinterpret_cast<const char*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
}  // namespace

TEST(SingleFileStorageSecurity, BlobLengthOverflowRejected) {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "ov_sfs_overflow_repro.bin";

    std::vector<char> buf;
    // Version header (major, minor, patch) matching SingleFileStorage::m_version {0,1,0}.
    put<uint16_t>(buf, 0);
    put<uint16_t>(buf, 1);
    put<uint16_t>(buf, 0);

    // TODO: a single backward-seek Blob record alone makes build_content_index loop/
    //       re-parse; a faithful PoC needs a preceding skipped record (>=256 bytes)
    //       so seekg(-256) lands inside the file and the scan still terminates true.
    //       Fill that in to exercise the full :118 -> :259 propagation.
    // Malicious Blob TLV: tag = Tag::Blob (0x03, uint32), length = 0xFFFFFFFFFFFFFF10.
    put<uint32_t>(buf, static_cast<uint32_t>(SingleFileStorage::Tag::Blob));
    put<uint64_t>(buf, 0xFFFFFFFFFFFFFF10ULL);  // length > INT64_MAX -> negative streamoff at line 112
    put<uint64_t>(buf, /*BlobId*/ 1ULL);
    put<uint64_t>(buf, /*PadSize*/ 0ULL);

    {
        std::ofstream os(path, std::ios::binary);
        os.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }

    SingleFileStorage storage(path);
    // Pre-fix: initialize() succeeds and m_blob_index[1].size == ~UINT64_MAX, then
    //   read_cache_entry triggers an invalid shape dimension in read_tensor_data.
    // Post-fix: the oversized residual is rejected, so loading the cache throws
    //   (OPENVINO_ASSERT at single_file_storage.cpp:346) instead of poisoning the index.
    EXPECT_THROW(
        {
            storage.initialize();
            storage.read_cache_entry("1", /*enable_mmap=*/true,
                                     [](auto&) {});
        },
        ov::Exception);

    std::error_code ec;
    fs::remove(path, ec);
}
