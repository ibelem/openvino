# Security finding #82: At lines 615-617, arguments_desc_size is read from the untrusted bl…

**Summary:** At lines 615-617, arguments_desc_size is read from the untrusted bl…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Denial-of-service via std::bad_alloc when loading a crafted cache blob with inflated arguments_desc_size or scalars_desc_size. When combined with the seek_current_ptr stream-cursor manipulation (finding 1), the resize can succeed (moderate N) while the subsequent loop reads cldnn::argument_desc::Types and cldnn::scalar_desc::Types from attacker-controlled blob bytes, enabling type confusion in GPU kernel argument dispatch and potential escalation to arbitrary GPU memory writes.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/src/kernel_selector/kernel_selector_common.cpp:615` — `clKernelData::load()`
**Validated for repos:** openvino
**Trust boundary:** GPU model cache blob loaded from filesystem or OpenVINO model-caching API into cldnn::BinaryInputBuffer

## Description / Root cause
At lines 615-617, arguments_desc_size is read from the untrusted blob and immediately passed to params.arguments.resize(arguments_desc_size) without any upper-bound check. Similarly at lines 622-624, scalars_desc_size is passed to params.scalars.resize(scalars_desc_size) without bounds validation. If the blob is crafted with a large size (e.g., 2^30), resize() either throws std::bad_alloc (DoS) or — when stream position is already corrupted by a prior seek_current_ptr exploit — succeeds with a moderate N and then the for-loop at lines 618-620 reads N * sizeof(argument_desc) bytes from the wrong stream position, populating cldnn::argument_desc::Types values with attacker-controlled data.

**Validator analysis:** The cited code is genuinely unbounded: arguments_desc_size/scalars_desc_size/local_memory_args_size are size_t values read straight from an untrusted blob and fed to std::vector::resize() with no sanity cap (lines 615-634). That matches CWE-789 (Memory Allocation with Excessive Size). The accurate impact is a denial-of-service: resize() of a 2^30 count throws std::length_error/std::bad_alloc after attempting a multi-GB allocation. The finding's escalation to 'type confusion' and 'arbitrary GPU memory writes' is NOT substantiated by this code alone — after resize(N) the loop at 618-620 reads exactly N elements into a buffer it just allocated, so there is no out-of-bounds access here; that escalation is entirely contingent on the separate, unproven 'finding 1' seek_current_ptr exploit and should be treated as speculation, not as this finding's impact. Note also a likely partial mitigation: GPU cache deserialization is normally wrapped so a load exception invalidates the cache and recompiles, downgrading this to a transient allocation spike rather than a hard crash — worth confirming, but the missing bound is still a real defect. The proposed fix (OPENVINO_ASSERT(size <= MAX) before each resize, applied to arguments, scalars, local_memory_args, and n_microkernels) is correct and sufficient for the DoS; pick the cap from the largest count any real kernel emits with generous headroom rather than a hardcoded 1024, and ensure the assert/throw is caught by the cache-load fallback so a bad blob just triggers recompilation.

## Exploit / Proof of Concept
Craft a cache blob where the arguments_desc_size field is set to 0x10000000 (268 million). When clKernelData::load() reads this value and calls params.arguments.resize(0x10000000), the process attempts to allocate ~2GB (if sizeof(argument_desc)=8), throws std::bad_alloc, and crashes the inference engine. For the type-confusion variant, set arguments_desc_size to a moderate value (e.g., 100) and corrupt the preceding stream cursor (via finding 1) so the loop at lines 618-620 reads attacker-controlled bytes as cldnn::argument_desc::Types values, which are then used verbatim to index GPU kernel argument slots.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_gpu_unit_tests (ASan build). Run: ov_gpu_unit_tests --gtest_filter=kernel_selector_clKernelData_load.rejects_excessive_arguments_desc_size . Pre-fix expectation: AddressSanitizer/allocator 'out of memory' or an uncaught std::bad_alloc/std::length_error from std::vector::resize at kernel_selector_common.cpp:617; post-fix expectation: a controlled ov::Exception from the added OPENVINO_ASSERT bound check, no allocation attempt.

## Suggested fix
Add a maximum bound check before each resize. Define a reasonable maximum (e.g., const size_t MAX_KERNEL_ARGS = 1024): 'OPENVINO_ASSERT(arguments_desc_size <= MAX_KERNEL_ARGS, "[GPU] arguments_desc_size exceeds maximum allowed value"); params.arguments.resize(arguments_desc_size);'. Apply the same guard to scalars_desc_size, local_memory_args_size, and n_microkernels. The bound should be derived from the maximum number of kernel arguments/scalars that any supported GPU kernel actually uses (typically < 64 arguments and < 32 scalars).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #82.
