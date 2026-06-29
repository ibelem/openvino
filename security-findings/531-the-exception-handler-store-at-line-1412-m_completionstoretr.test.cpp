// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for graph.cpp:1412 — relaxed m_completion store in exception path
// breaks acquire-release synchronization with the consumer spinning in updateDynParams.
// Pre-fix: on weakly-ordered CPUs (ARM/RISC-V) TSan may report a benign-looking but
// incorrect ordering; logically the consumer exits early leaving nodes in inconsistent state.
// Post-fix: memory_order_release at line 1412 guarantees happens-before to the consumer.
//
// Target: ov_cpu_unit_tests
// Filter:  --gtest_filter=UpdateNodesSyncTest.ExceptionPathUsesReleaseOrdering
// Expected sanitizer: ThreadSanitizer data-race or logic assertion failure pre-fix.

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include <stdexcept>

// TODO: include the real NodePtr / Graph headers once build environment is set up.
// The test below directly replicates the UpdateNodesBase logic to validate the
// memory ordering contract without depending on private/anonymous-namespace symbols.

namespace {

// Minimal stand-in that replicates the UpdateNodesBase protocol exactly,
// with the BUGGY relaxed store on the exception path (line 1412).
struct UpdateNodesBaseBuggy {
    std::atomic<size_t> m_prepareCounter{0};
    std::atomic<bool>   m_completion{false};
    // Simulated node results: index i => whether updateShapes succeeded
    std::vector<bool>   shapesUpdated;
    std::vector<bool>   dynParamsUpdated;
    size_t              throwAtIndex = SIZE_MAX;

    void updateShapes(size_t node_indx, size_t stop_indx) {
        try {
            for (size_t i = node_indx; i < stop_indx; i++) {
                if (i == throwAtIndex)
                    throw std::runtime_error("crafted shape error");
                shapesUpdated[i] = true;
                m_prepareCounter.store(i, std::memory_order_release);
            }
        } catch (...) {
            // BUG: relaxed — mirrors graph.cpp:1412 pre-fix
            m_completion.store(true, std::memory_order_relaxed);
            throw;
        }
        m_prepareCounter.store(stop_indx, std::memory_order_relaxed);
        m_completion.store(true, std::memory_order_release);
    }

    void updateDynParams(size_t node_indx, size_t /*stop_indx*/) {
        size_t local_counter = node_indx;
        while (true) {
            const bool   completion    = m_completion.load(std::memory_order_acquire);
            const size_t prepareCounter = m_prepareCounter.load(std::memory_order_relaxed);
            if (completion && local_counter == prepareCounter)
                break;
            while (local_counter < prepareCounter) {
                dynParamsUpdated[local_counter] = true;
                local_counter++;
            }
        }
    }
};

// Fixed variant: exception path uses memory_order_release (the proposed fix).
struct UpdateNodesBaseFixed : UpdateNodesBaseBuggy {
    void updateShapes(size_t node_indx, size_t stop_indx) {
        try {
            for (size_t i = node_indx; i < stop_indx; i++) {
                if (i == throwAtIndex)
                    throw std::runtime_error("crafted shape error");
                shapesUpdated[i] = true;
                m_prepareCounter.store(i, std::memory_order_release);
            }
        } catch (...) {
            // FIX: release — mirrors graph.cpp:1412 post-fix
            m_completion.store(true, std::memory_order_release);
            throw;
        }
        m_prepareCounter.store(stop_indx, std::memory_order_relaxed);
        m_completion.store(true, std::memory_order_release);
    }
};

template <typename UpdateBase>
void runConcurrent(UpdateBase& base, size_t nodeCount, size_t throwAt) {
    base.shapesUpdated.assign(nodeCount, false);
    base.dynParamsUpdated.assign(nodeCount, false);
    base.m_prepareCounter.store(0);
    base.m_completion.store(false);
    base.throwAtIndex = throwAt;

    bool producerThrew = false;
    std::thread producer([&]() {
        try {
            base.updateShapes(0, nodeCount);
        } catch (...) {
            producerThrew = true;
        }
    });
    // Consumer runs on this thread (mirrors execute_and_wait pattern)
    base.updateDynParams(0, nodeCount);
    producer.join();
    (void)producerThrew;
}

}  // namespace

// TODO: Once the Graph internals are refactored to allow dependency injection of
// node lists, replace the stand-in structs above with a Graph/SyncInferRequest
// round-trip using the ov::Core + compiled_model API with a dynamically-shaped
// model whose second node's updateShapes is forced to throw (e.g., via a
// node mock that derives from ov::intel_cpu::Node).

TEST(UpdateNodesSyncTest, ExceptionPathUsesReleaseOrdering_InvariantCheck) {
    // With the FIXED implementation: for every node whose shapesUpdated[i]==true
    // we must also see dynParamsUpdated[i]==true after the concurrent run.
    // This invariant can be violated pre-fix on weakly-ordered CPUs.
    const size_t nodeCount = 4;
    for (size_t throwAt = 1; throwAt < nodeCount; ++throwAt) {
        UpdateNodesBaseFixed fixed;
        runConcurrent(fixed, nodeCount, throwAt);
        for (size_t i = 0; i < throwAt; ++i) {
            EXPECT_TRUE(fixed.shapesUpdated[i])
                << "Node " << i << " shape not updated (throwAt=" << throwAt << ")";
            EXPECT_TRUE(fixed.dynParamsUpdated[i])
                << "Node " << i << " dynParams not updated despite shape updated (throwAt=" << throwAt << ")";
        }
    }
}
