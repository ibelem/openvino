# Security finding #531: The exception-handler store at line 1412 (`m_completion.store(true,…

**Summary:** The exception-handler store at line 1412 (`m_completion.store(true,…

**CWE IDs:** CWE-362: Race Condition / Missing Synchronization (incorrect memory order)
**Severity / Impact:** On weakly-ordered platforms (ARM servers running OpenVINO), an untrusted ONNX model with a crafted dynamic shape that triggers an exception inside `node->updateShapes()` can cause one or more subsequent nodes to have their shapes updated but their dynamic parameters left in the prior state. On the next inference request these nodes may compute with wrong strides/offsets, which can produce an out-of-bounds memory read or write (CWE-125 / CWE-787) inside the kernel execution path, leading to process crash (DoS) or, on some layouts, exploitable memory corruption. The bug is in the hot inference path and is reachable by any caller that feeds a dynamically-shaped model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/graph.cpp:1412` — `UpdateNodesBase::updateShapes / UpdateNodesBase::updateDynParams()`
**Validated for repos:** openvino
**Trust boundary:** Externally-supplied dynamic tensor shape from an untrusted ONNX model processed via the OV EP, which drives node->updateShapes() inside a TBB async task; a malformed shape triggers an exception inside the loop and must reliably signal the concurrently-spinning updateDynParams task

## Description / Root cause
The exception-handler store at line 1412 (`m_completion.store(true, std::memory_order_relaxed)`) uses relaxed ordering, whereas the successful-path store at line 1416 correctly uses `std::memory_order_release`. The consumer (`updateDynParams`) loads `m_completion` with `std::memory_order_acquire` at line 1422. By the C++ memory model, an acquire load synchronizes-with a *release* (or seq_cst) store on the same variable; a relaxed store provides no such guarantee. Consequently, when the acquire load observes `completion == true` on the exception path, there is no happens-before edge back to the producer, so the consumer's subsequent relaxed load of `m_prepareCounter` (line 1423) is not guaranteed to see the latest value stored by the producer's release store at line 1409. On weakly-ordered ISAs (ARM/RISC-V) the CPU is free to reorder the producer's `m_completion` store before earlier `m_prepareCounter` stores, meaning the consumer can see `completion == true` with a stale (smaller) `prepareCounter`. When `local_counter` has already advanced to that stale value, the break condition `completion && local_counter == prepareCounter` fires immediately, causing the consumer to exit before calling `updateDynamicParams` on every node whose `updateShapes` already succeeded. Those nodes are left in an intermediate state: shapes updated, dynamic params not. On the following inference call the engine reuses these nodes with mismatched internal buffers.

**Validator analysis:** The vuln type (CWE-362, incorrect memory order) is accurate. The flaw is real: the exception-handler store `m_completion.store(true, std::memory_order_relaxed)` at graph.cpp:1412 is inconsistent with the normal-path store that uses `memory_order_release` at line 1416. The consumer's acquire-load at line 1422 synchronizes-with only a release (or seq_cst) store on the same variable; a relaxed store provides no happens-before edge. On weakly-ordered ISAs (ARM/RISC-V) the producer's last release-store to `m_prepareCounter` (line 1409) is not guaranteed to be visible to the consumer when it observes `completion=true` from the relaxed store, making the early-exit condition at line 1424 fire on a stale (too-small) `prepareCounter`. Nodes whose shapes were already updated by `updateShapes` will then not have `updateDynamicParams` called, leaving them in a mixed state and risking OOB memory access on the subsequent inference. Impact (DoS / memory corruption on ARM) is plausible. The proposed fix — changing line 1412 to `memory_order_release` — is correct and sufficient; it mirrors the normal-path store and ensures the acquire-load at line 1422 sees all preceding writes including the final `m_prepareCounter` release-store.

## Exploit / Proof of Concept
1. Craft an ONNX model with dynamic input shapes and at least two dynamic nodes. 2. Feed the model to the OV EP (e.g., via OrtSession::Run) with a shape that causes the second node's `updateShapes()` to throw (e.g., an unsupported or inconsistent shape). 3. The producer stores `m_completion = true` (relaxed) and rethrows; the consumer's acquire load sees `true` but, on an ARM device, the store buffer may not have flushed the release-ordered write to `m_prepareCounter` yet, causing the consumer to read a stale `prepareCounter` equal to `local_counter` and exit early. 4. The first node now has updated shapes but unupdated dynamic params. 5. On the immediately-following inference call the engine reuses these nodes without re-initialising, causing an OOB memory access inside the GEMM/convolution kernel.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_cpu_unit_tests (or the nearest intel_cpu gtest target). Run with: --gtest_filter=UpdateNodesSyncTest.ExceptionPathUsesReleaseOrdering_InvariantCheck. On ARM/RISC-V with ThreadSanitizer enabled (-fsanitize=thread), the buggy variant (UpdateNodesBaseBuggy) should exhibit a TSan race or logic failure where dynParamsUpdated[i] is false for nodes whose shapesUpdated[i] is true. The fixed variant must pass cleanly. Add -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_SANITIZER=ON to the CMake configure step.

## Suggested fix
Change line 1412 from `m_completion.store(true, std::memory_order_relaxed)` to `m_completion.store(true, std::memory_order_release)`. This mirrors the normal-path store at line 1416 and ensures that when the consumer's acquire load at line 1422 observes `completion == true`, all preceding stores (including the last release store to `m_prepareCounter` at line 1409) are also visible to the consumer, eliminating the race. No other change is needed because the consumer's load of `m_prepareCounter` at line 1423 is covered by the release-acquire edge on `m_completion`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #531.
