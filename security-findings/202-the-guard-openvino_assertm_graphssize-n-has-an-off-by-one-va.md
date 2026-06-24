# Security finding #202: The guard `OPENVINO_ASSERT(m_graphs.size() >= n, ...)` has an off-b…

**Summary:** The guard `OPENVINO_ASSERT(m_graphs.size() >= n, ...)` has an off-b…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** Undefined behaviour / OOB read on m_graphs. In optimized builds std::vector::operator[] performs no bounds check, so the read returns a garbage shared_ptr. Dereferencing it in the callers (get_runtime_model, SyncInferRequest constructor, etc.) results in a crash or, in adversarial scenarios, type confusion / memory disclosure. Affects every inference path that creates a SyncInferRequest or calls get_runtime_model when num_streams=0.
**Affected location:** `targets/openvino/src/plugins/intel_gpu/src/plugin/compiled_model.cpp:245` — `CompiledModel::get_graph()`
**Validated for repos:** openvino
**Trust boundary:** Stream-id value (n) passed by callers vs. actual m_graphs population; also: model/cache deserialization path where m_config.get_num_streams() may return 0, leaving m_graphs empty

## Description / Root cause
The guard `OPENVINO_ASSERT(m_graphs.size() >= n, ...)` has an off-by-one: valid indices are 0..size-1, requiring `size > n`, but `>=` allows n == size(). When m_graphs is empty (size==0) and the universal call site uses n=0, `0 >= 0` is true, the assertion does not fire, and `m_graphs[0]` on line 246 reads past the end of an empty vector (UB / OOB read).

**Validator analysis:** The off-by-one is real: valid indices are 0..size-1, so the guard must be `size > n`, but `>=` permits n==size. The only way m_graphs is empty is num_streams==0 (ctor loops at compiled_model.cpp:65 and :160 push nothing). num_streams is a user-settable RELEASE option (options.inl:9) with NO validator, and apply_performance_hints (execution_config.cpp:406-433) only special-cases AUTO(-1), never normalizing 0 to 1, so get_num_streams() can return 0 after finalize. Then get_runtime_model()/SyncInferRequest call get_graph(0); `0>=0` passes and m_graphs[0] is an OOB read on an empty vector — UB, garbage shared_ptr deref. So CWE-125 is accurate for the read itself. However the impact text overstates it: this is a robustness/crash bug from a misconfiguration, not a realistic 'memory disclosure / type confusion' from adversarial input — num_streams=0 is operator error, and the deserialization ctor takes num_streams from m_config (a runtime-supplied config), NOT from the cache blob, so the 'crafted blob → stream count 0' exploit hypothesis is unfounded. The proposed fix (change `>=` to `>`) is correct and sufficient to convert the OOB into a clean OPENVINO_ASSERT throw; the suggested extra `!m_graphs.empty()` guards in get_runtime_model/SyncInferRequest are nice-to-have defense-in-depth but redundant once the index check is strict, since get_graph(0) on an empty vector would then throw 'Invalid graph idx: 0. Only 0 were created'. Best fix: use `>` (and optionally normalize num_streams==0 to 1 at finalize, matching CPU plugin semantics).

## Exploit / Proof of Concept
Construct a CompiledModel with a config where get_num_streams() returns 0 (e.g., explicitly set via ov::num_streams(0) or via a crafted serialized blob that causes the stream count to be 0 after deserialization). The constructor loops `for (uint16_t n = 0; n < 0; n++)` and pushes nothing, leaving m_graphs empty. Any subsequent call to get_runtime_model() or creation of a SyncInferRequest calls get_graph(0); the condition `0 >= 0` is true so the OPENVINO_ASSERT does not fire, then `m_graphs[0]` on the empty vector is an OOB read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the off-by-one at
//   openvino/src/plugins/intel_gpu/src/plugin/compiled_model.cpp:245-246
// Pre-fix: with ov::num_streams(0) the CompiledModel ctor leaves m_graphs empty,
//   and get_graph(0) does `m_graphs[0]` on an empty vector (OOB read; ASan: heap/container-overflow).
// Post-fix (`m_graphs.size() > n`): get_graph(0) throws ov::Exception ("Invalid graph idx: 0").
// Target: ov_gpu_unit_tests (gtest + ASan).
//
// NOTE (skeleton): a real CompiledModel needs a live GPU engine/RemoteContextImpl and a
// finalized ExecutionConfig; building one inline is non-trivial in the unit harness.
// Fill the TODOs using the existing fixtures under intel_gpu/tests/unit/.
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

using namespace ov::intel_gpu;

TEST(compiled_model_get_graph, num_streams_zero_must_not_oob) {
    // TODO: obtain a GPU engine + RemoteContextImpl::Ptr from the unit fixtures
    //       (see tests that already build a cldnn::engine / RemoteContextImpl).
    // auto engine = ...; auto ctx = std::make_shared<RemoteContextImpl>(...);

    ExecutionConfig cfg;
    cfg.set_property(ov::num_streams(0));   // unvalidated -> get_num_streams()==0
    // TODO: cfg.finalize(*engine);          // must keep num_streams==0 (no AUTO normalization)

    // TODO: build a trivial ov::Model (e.g. Parameter->Result) and the plugin ptr.
    // auto model = ...; auto plugin = ...;
    // CompiledModel cm(model, plugin, ctx, cfg);   // ctor loop pushes nothing -> m_graphs empty

    // The contract: get_graph(0) / get_runtime_model() on an empty m_graphs MUST throw,
    // not read out of bounds. Pre-fix this line is an OOB read (ASan abort); post-fix it throws.
    // EXPECT_THROW(cm.get_runtime_model(), ov::Exception);
    GTEST_SKIP() << "TODO: wire GPU engine/context/model fixtures, then remove skip.";
}
```
**Build / run:** Build: cmake --build . --target ov_gpu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ./ov_gpu_unit_tests --gtest_filter=compiled_model_get_graph.num_streams_zero_must_not_oob . Pre-fix expectation: AddressSanitizer 'container-overflow'/'heap-buffer-overflow READ' at compiled_model.cpp:246 (m_graphs[0] on empty vector). Post-fix expectation: ov::Exception thrown with message 'Invalid graph idx: 0. Only 0 were created' and the test passes.

## Suggested fix
Change the assertion to use strictly-greater-than: `OPENVINO_ASSERT(m_graphs.size() > n, "[GPU] Invalid graph idx: ", n, ". Only ", m_graphs.size(), " were created");`. Additionally add an `OPENVINO_ASSERT(!m_graphs.empty(), ...)` guard at the top of get_runtime_model() and a matching check in SyncInferRequest::SyncInferRequest before calling get_graph(0), analogous to the existing guard in export_model (line 182).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #202.
