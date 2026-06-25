# Security finding #358: Line 104: `const auto base_dir = base.empty() ? std::filesystem::cu…

**Summary:** Line 104: `const auto base_dir = base.empty() ? std::filesystem::cu…

**CWE IDs:** CWE-73: External Control of File Name or Path
**Severity / Impact:** An attacker supplying a crafted ONNX model loaded from memory (no file path) can embed any relative path in external_data.location (e.g., 'config/secrets.json', 'ssl/private.key') to read arbitrary files that happen to reside within or below the process's CWD. If CWD is the filesystem root, application install directory, or another broad location, this is an arbitrary local file read.
**Affected location:** `targets/openvino/src/common/util/src/file_util.cpp:104` — `ov::util::sanitize_path()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled ONNX proto external_data 'location' field → m_data_location → sanitize_path(model_dir, make_path(m_data_location)); model_dir can be empty when the model is loaded from a memory buffer or when get_model_dir() returns {} (tensor.cpp:35)

## Description / Root cause
Line 104: `const auto base_dir = base.empty() ? std::filesystem::current_path() : base;`. When model_dir is empty (model loaded from memory with no file path, or dynamic_cast failure in get_model_dir()), sanitize_path silently substitutes CWD as the containment boundary. The `..` escape check still applies (so an attacker cannot go ABOVE CWD), but any relative path that stays within CWD subtree is permitted, regardless of whether it is inside the intended model directory. The security invariant—external data must reside within the model's directory—is completely lost.

**Validator analysis:** The cited flaw is real and reachable in openvino: TensorExternalData::load_external_data/load_external_mmap_data (tensor_external_data.cpp:47,77) pass the attacker-controlled external_data 'location' (m_data_location, set at line 22) and a possibly-empty model_dir into ov::util::sanitize_path. When model_dir is empty — exactly the case for models read from a memory buffer where get_model_dir() returns {} (tensor.cpp:35) — sanitize_path falls back to current_path() (file_util.cpp:104) as the containment base, so the model-directory invariant is replaced with the process CWD. CWE-73 is the correct category. HOWEVER the stated impact ('arbitrary local file read') is overstated: line 108's lexically_relative '..' check is a present, working mitigation that blocks escaping the CWD subtree, and absolute-path locations that point outside CWD also get a leading '..' and are rejected. The only true arbitrary-read case is the degenerate CWD='/' deployment the finding itself describes; otherwise the exposure is bounded to files under the process working directory, and exfiltration requires the loaded bytes to be observable via model outputs. So this is a genuine but lower-severity containment weakening, not unbounded arbitrary read. The proposed fix is reasonable: option (a) — have load_external_data/load_external_mmap_data throw error::invalid_external_data when model_dir.empty() (in-memory models have no legitimate external-data directory) — is the cleanest and sufficient; rejecting relative_path.is_absolute() before operator/ is a good defense-in-depth addition. Documenting the CWD fallback alone is NOT sufficient.

## Exploit / Proof of Concept
Construct an ONNX proto with a TensorProto whose external_data[location] = 'etc/passwd' (or any path relative to CWD). Call core.read_model() from a byte buffer (no file path). get_model_dir() returns empty; sanitize_path falls back to CWD; the path 'etc/passwd' resolves to CWD/etc/passwd. If CWD = '/', the check at line 108 computes lexically_relative('/etc/passwd', '/') = 'etc/passwd' (does NOT begin with '..', is NOT empty), so the exception is not thrown and the file is opened and returned as tensor data.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for file_util.cpp:104 / tensor_external_data.cpp:47,77 (work item 358):
// when an ONNX model is parsed from memory (empty model_dir) and a tensor declares
// external_data with a relative 'location', the loader MUST reject it instead of
// silently resolving against std::filesystem::current_path().
// Pre-fix: sanitize_path falls back to CWD and the read succeeds (no throw),
//          confirming loss of the model-dir containment invariant.
// Post-fix: load_external_data throws ov::Exception / invalid_external_data.
//
// Target test binary: ov_onnx_frontend_tests (gtest), style of onnx_import.in.cpp.
//
// NOTE: this needs a crafted .onnx whose initializer uses data_location=EXTERNAL
// and external_data[location] pointing at a file under CWD, fed through an
// in-memory stream (no model path) so get_model_dir() == {}. The model bytes are
// not available here, so this is a SKELETON.

#include <fstream>
#include <sstream>
#include "common_test_utils/test_assertions.hpp"
#include "onnx_utils.hpp"  // FrontEndTestUtils / load_from_stream helpers

using namespace ov::frontend::onnx::tests;

TEST(onnx_importer_external_data, reject_external_data_when_model_dir_empty) {
    // TODO: replace with a real serialized ONNX proto (in-memory, no file path) whose
    //       single initializer has data_location = EXTERNAL and
    //       external_data = { {"location", "some_relative_file.bin"}, {"length","4"} }.
    //       Build it via onnx::ModelProto and SerializeToString so model_dir is empty.
    std::string model_bytes = /* TODO: crafted_external_data_model() */ "";
    ASSERT_FALSE(model_bytes.empty()) << "TODO: supply crafted in-memory ONNX model";

    std::istringstream model_stream(model_bytes);

    // TODO: confirm the exact helper that reads a model from a stream with NO path
    //       (e.g. ov::Core::read_model(stream) or FrontEnd::load(stream)) in this tree.
    // Pre-fix this either succeeds or reads CWD/some_relative_file.bin; post-fix it
    // must throw because model_dir is empty for an in-memory model.
    OV_EXPECT_THROW(/* auto model = */ load_model_from_stream(model_stream),
                    ov::Exception,
                    testing::HasSubstr("external data"));
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests . Run: ./ov_onnx_frontend_tests --gtest_filter=onnx_importer_external_data.reject_external_data_when_model_dir_empty . Pre-fix expectation: no throw (file under CWD is opened/read), so the test FAILS; post-fix expectation: load_external_data throws ov::Exception/invalid_external_data when model_dir is empty, so the test PASSES. (Needs a crafted in-memory .onnx fixture — see TODOs.)

## Suggested fix
When base is empty (model loaded without a file path), either (a) raise an error immediately — external data is not supported for in-memory models — or (b) require that external data paths are explicitly whitelisted. At minimum, document the CWD-fallback behavior and add a caller-side check in load_external_data / load_external_mmap_data: if model_dir.empty(), throw error::invalid_external_data before calling sanitize_path. Additionally, reject relative_path values that are themselves absolute paths explicitly before calling operator/ (check relative_path.is_absolute()).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #358.
