// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/tensor_external_data.hpp"

#include <fstream>
#include <sstream>

#include "exceptions.hpp"
#include "openvino/runtime/lazy_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"

#include <cstdint>
#ifdef _WIN32
#    include <windows.h>
#else
#    include <cstdio>
#endif

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {
namespace {
// Validate that [ptr, ptr + size) refers to a currently mapped, readable region of this
// process's address space. Used to guard the ORT shared-memory interop path against an
// attacker-controlled offset that is reinterpreted as a raw pointer.
bool is_readable_memory_region(const void* ptr, size_t size) {
    if (size == 0) {
        return true;
    }
    if (ptr == nullptr) {
        return false;
    }
#if defined(_WIN32)
    const char* p = static_cast<const char*>(ptr);
    const uintptr_t start_addr = reinterpret_cast<uintptr_t>(p);
    const uintptr_t end_addr = start_addr + size;
    uintptr_t cur = start_addr;
    while (cur < end_addr) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(reinterpret_cast<const void*>(cur), &mbi, sizeof(mbi)) == 0) {
            return false;
        }
        if (mbi.State != MEM_COMMIT) {
            return false;
        }
        const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                               PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY;
        if ((mbi.Protect & readable) == 0) {
            return false;
        }
        if (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) {
            return false;
        }
        const uintptr_t region_end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        if (region_end <= cur) {
            return false;
        }
        cur = region_end;
    }
    return true;
#elif defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) {
        return false;
    }
    const uintptr_t start_addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end_addr = start_addr + size;
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long region_start = 0;
        unsigned long long region_end = 0;
        char perms[5] = {0};
        if (std::sscanf(line.c_str(), "%llx-%llx %4s", &region_start, &region_end, perms) >= 3) {
            if (region_start <= start_addr && end_addr <= region_end) {
                return perms[0] == 'r';
            }
        }
    }
    return false;
#else
    // Unknown platform: be conservative and reject raw-address interop.
    return false;
#endif
}
}  // namespace

TensorExternalData::TensorExternalData(const TensorProto& tensor) {
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") {
            m_data_location = entry.value();
        } else if (entry.key() == "offset") {
            m_offset = std::stoull(entry.value());
        } else if (entry.key() == "length") {
            m_data_length = std::stoull(entry.value());
        } else if (entry.key() == "checksum") {
            m_sha1_digest = entry.value();
        }
    }
#ifdef ENABLE_OPENVINO_DEBUG
    if (m_sha1_digest.size() > 0) {
        OPENVINO_WARN("SHA1 checksum is not supported");
    }
#endif
}
TensorExternalData::TensorExternalData(const std::string& location, size_t offset, size_t size) {
    m_data_location = location;
    m_offset = offset;
    m_data_length = size;
}

Buffer<ov::MappedMemory> TensorExternalData::load_external_mmap_data(const std::filesystem::path& model_dir,
                                                                     MappedMemoryHandles cache) const {
    std::filesystem::path full_path;
    try {
        full_path = ov::util::sanitize_path(model_dir, ov::util::make_path(m_data_location));
    } catch (const std::runtime_error& e) {
        throw error::invalid_external_data{e.what()};
    }

    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || m_data_length > static_cast<uint64_t>(file_size) ||
        m_offset > static_cast<uint64_t>(file_size) - m_data_length) {
        throw error::invalid_external_data{*this};
    }
    auto cached_mapped_memory = cache->find(full_path);
    std::shared_ptr<ov::MappedMemory> mapped_memory;
    if (cached_mapped_memory != cache->end()) {
        mapped_memory = cached_mapped_memory->second;
    } else {
        mapped_memory = ov::load_mmap_object(full_path);
        (*cache)[full_path] = mapped_memory;
    }
    if (m_data_length > mapped_memory->size() || mapped_memory->size() == 0) {
        throw error::invalid_external_data{*this};
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data() + m_offset,
        m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset,
        mapped_memory);
}

Buffer<ov::AlignedBuffer> TensorExternalData::load_external_data(const std::filesystem::path& model_dir) const {
    std::filesystem::path full_path;
    try {
        full_path = ov::util::sanitize_path(model_dir, ov::util::make_path(m_data_location));
    } catch (const std::runtime_error& e) {
        throw error::invalid_external_data{e.what()};
    }

    const auto file_size = util::file_size(full_path);
    if (file_size < 0 || m_data_length > static_cast<uint64_t>(file_size) ||
        m_offset > static_cast<uint64_t>(file_size) - m_data_length) {
        throw error::invalid_external_data{*this};
    }

    uint64_t read_data_length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;
    const auto get_now_buffer = [&]() {
        std::ifstream external_data_stream(full_path, std::ios::binary | std::ios::in | std::ios::ate);
        if (external_data_stream.fail()) {
            throw error::invalid_external_data{*this};
        }
        // default value of m_offset is 0
        external_data_stream.seekg(m_offset, std::ios::beg);

        auto read_data = std::make_shared<ov::AlignedBuffer>(read_data_length);
        external_data_stream.read(read_data->get_ptr<char>(), read_data_length);
        external_data_stream.close();
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(read_data->get_ptr<char>(),
                                                                                      read_data->size(),
                                                                                      read_data);
    };
    const auto get_lazy_buffer = [&]() {
        const auto lazy = std::make_shared<LazyBuffer>(full_path, m_offset, read_data_length);
        return std::make_shared<SharedBuffer<std::shared_ptr<AlignedBuffer>>>(
            static_cast<char*>(lazy->get_reserved_ptr()),
            lazy->size(),
            lazy);
    };

    constexpr size_t lazy_loading_threshold = 0x100000;  // 1MB
    return read_data_length >= lazy_loading_threshold ? get_lazy_buffer() : get_now_buffer();
}

Buffer<ov::AlignedBuffer> TensorExternalData::load_external_mem_data() const {
    if (m_data_location != ORT_MEM_ADDR) {
        throw error::invalid_external_data{*this};
    }
    // Empty node will create a constant with zero shape and zero size external data.
    bool is_valid_buffer = m_offset && m_data_length;
    bool is_empty_buffer = (m_data_length == 0);
    if (!(is_valid_buffer || is_empty_buffer)) {
        throw error::invalid_external_data{*this};
    }
    char* addr_ptr = reinterpret_cast<char*>(m_offset);
    // m_offset is reinterpreted as a raw in-process address. Before copying, verify that the
    // requested range is actually mapped and readable so an attacker-controlled offset coming
    // from an ONNX model cannot trigger an out-of-bounds read of arbitrary process memory.
    if (m_data_length > 0 && !is_readable_memory_region(addr_ptr, m_data_length)) {
        throw error::invalid_external_data{*this};
    }
    auto aligned_memory = std::make_shared<ov::AlignedBuffer>(m_data_length);
    if (m_data_length > 0) {
        std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length);
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(aligned_memory->get_ptr<char>(),
                                                                                  aligned_memory->size(),
                                                                                  aligned_memory);
}

std::string TensorExternalData::to_string() const {
    std::stringstream s;
    s << "ExternalDataInfo(";
    s << "data_full_path: " << m_data_location;
    s << ", offset: " << m_offset;
    s << ", data_length: " << m_data_length;
    if (m_sha1_digest.size() > 0) {
        s << ", sha1_digest: " << m_sha1_digest << ")";
    } else {
        s << ")";
    }
    return s.str();
}
}  // namespace detail
}  // namespace onnx
}  // namespace frontend
}  // namespace ov