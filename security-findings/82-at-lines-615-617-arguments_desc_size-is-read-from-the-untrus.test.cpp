// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-789 at
//   openvino/src/plugins/intel_gpu/src/kernel_selector/kernel_selector_common.cpp:615-624
// Pre-fix: clKernelData::load() calls params.arguments.resize(<attacker size>) with no
//          upper bound -> ~GB allocation attempt -> std::bad_alloc / std::length_error.
// Post-fix: load() validates the count and throws a controlled ov::Exception
//           (OPENVINO_ASSERT) BEFORE attempting the allocation.
// SKELETON: the exact BinaryInputBuffer construction-from-bytes API and the
// kernel_selector::clKernelData accessor visibility must be confirmed against the
// intel_gpu test tree before this compiles.

#include <gtest/gtest.h>
// TODO: confirm include paths from intel_gpu/tests/unit/ — e.g.
//   #include "intel_gpu/runtime/engine.hpp"
//   #include "intel_gpu/graph/serialization/binary_buffer.hpp"   // cldnn::BinaryInputBuffer
//   #include "kernel_selector_common.h"                          // kernel_selector::clKernelData

// TODO: clKernelData / its params may be internal to the kernel_selector static lib;
//   if not linkable from ov_gpu_unit_tests, drive this through the public kernel-cache
//   deserialization entry point instead (kernels_cache load) with a crafted blob.

TEST(kernel_selector_clKernelData_load, rejects_excessive_arguments_desc_size) {
    // Build a serialized blob whose arguments_desc_size field is a huge value (0x10000000).
    // Layout mirrors clKernelData::save(): workGroups.global, workGroups.local,
    // then size_t arguments_desc_size.
    std::vector<uint8_t> blob;
    // TODO: append the exact bytes for params.workGroups.global / .local as written by save().
    const uint64_t huge_arguments_desc_size = 0x10000000ULL; // 268M entries
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&huge_arguments_desc_size);
    blob.insert(blob.end(), p, p + sizeof(huge_arguments_desc_size));

    // TODO: wrap `blob` in the stream type cldnn::BinaryInputBuffer expects
    //   (membuf / istream). Use the same helper the existing serialization unit
    //   tests use under intel_gpu/tests/unit/.
    // cldnn::BinaryInputBuffer ib(stream, test_engine());
    // kernel_selector::clKernelData kd;

    // Pre-fix this either crashes the allocator or throws std::bad_alloc;
    // post-fix it must throw the controlled OPENVINO_ASSERT (ov::Exception).
    // ASSERT_THROW(kd.load(ib), ov::Exception);
    GTEST_SKIP() << "TODO: provide concrete BinaryInputBuffer construction + clKernelData linkage";
}