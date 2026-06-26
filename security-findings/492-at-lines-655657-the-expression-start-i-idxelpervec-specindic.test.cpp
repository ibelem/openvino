// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression target for gather.cpp:655-657 (uint64_t -> int truncation of
// srcBeforeAxisDiff when axisAndAfterAxisSizeInBytes > INT32_MAX).
// Harness: ov_cpu_unit_tests (gtest), test lives under
//   openvino/src/plugins/intel_cpu/tests/unit/
//
// SKELETON: a faithful end-to-end trigger needs a Gather data tensor whose
// axisDim*afterAxisSize*dataTypeSize > 2^31 (~2GB allocation) with
// afterAxisSize in [2..dataElPerVec], which is infeasible to allocate in a
// unit test. Encode the fix instead as a guard check on the computed scalar.
//
// TODO: replace the placeholder include/helper names with the real ones from
//       intel_cpu/tests/unit/ once confirmed by reading that tree.
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

namespace {
// Mirrors the offending expression's pre-store value (gather.cpp:656-657).
// k is the small quotient-difference (0/1) multiplied by the byte stride.
static int64_t computeBeforeAxisDiff(uint64_t k, uint64_t axisAndAfterAxisSizeInBytes) {
    return static_cast<int64_t>(k * axisAndAfterAxisSizeInBytes);
}
}

// Pre-fix: storing into std::vector<int> truncates; this test documents that
// the guard added by the fix must reject such a configuration.
TEST(GatherShortParams, SrcBeforeAxisDiffMustNotTruncateToInt) {
    // afterAxisSize=16 (fp32 blocked short case), axisDim chosen so the byte
    // stride exceeds INT32_MAX.
    const uint64_t afterAxisSize = 16;
    const uint64_t dataTypeSize  = 4; // f32
    const uint64_t axisDim       = 34000000ULL; // > 2^31 / (16*4)
    const uint64_t axisAndAfterAxisSizeInBytes = axisDim * afterAxisSize * dataTypeSize;

    ASSERT_GT(axisAndAfterAxisSizeInBytes,
              static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));

    const int64_t full = computeBeforeAxisDiff(/*k=*/1, axisAndAfterAxisSizeInBytes);
    const int truncated = static_cast<int>(full); // what gather.h:41 does today

    // Demonstrates the silent corruption the fix must prevent.
    EXPECT_NE(static_cast<int64_t>(truncated), full)
        << "srcBeforeAxisDiff truncated from " << full << " to " << truncated;

    // TODO: once the prepareParams guard
    //   CPU_NODE_ASSERT(axisAndAfterAxisSizeInBytes <= INT32_MAX, ...)
    // is added, replace the above with an end-to-end build of a Gather node
    // (Node::prepareParams) and EXPECT throw, e.g.:
    //   EXPECT_THROW(buildAndPrepareGather(axisDim, afterAxisSize, dataTypeSize), ov::Exception);
}
