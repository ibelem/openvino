# Security finding #394: `sanitize_path` at line 47 calls `std::filesystem::weakly_canonical…

**Summary:** `sanitize_path` at line 47 calls `std::filesystem::weakly_canonical…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** An attacker who can write to the model directory (world-writable temp directory, shared multi-tenant environment) can cause OpenVINO to map-read arbitrary files outside the intended model directory, exposing sensitive file contents (e.g. `/etc/shadow`, private keys). The same window exists in `load_external_data` (ifstream at line 90) and `extract_tensor_external_data` in `graph_iterator_proto.cpp` (lines 323, 339).
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:47` — `TensorExternalData::load_external_mmap_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model-supplied `location` string (attacker-controlled) in a shared/world-writable model directory → resolved path used for file open after check

## Description / Root cause
`sanitize_path` at line 47 calls `std::filesystem::weakly_canonical` which resolves symlinks and non-existent path tails **at check time** and returns a plain `std::filesystem::path` string. For any path component that does not yet exist at check time (e.g., the filename itself), `weakly_canonical` appends it verbatim without resolving. The returned `full_path` is then passed separately to `ov::util::file_size(full_path)` (line 52) and `ov::load_mmap_object(full_path)` (line 62) — two additional OS path-resolution calls that re-traverse the filesystem. No file descriptor is held atomically across the check and the open.

**Validator analysis:** The TOCTOU is real at code level: ov::util::sanitize_path (file_util.cpp:104-115) computes weakly_canonical(base/relative) and verifies lexically_relative does not escape base, then returns the resolved path STRING. weakly_canonical does not open a descriptor and, for a non-existent final component, appends it verbatim so a still-absent target passes containment. The returned full_path is independently re-traversed by ov::util::file_size (line 52) and ov::load_mmap_object (line 62) — and again by std::ifstream in load_external_data (line 90) — with no fd carried across check→use. An attacker who can write the model directory (world-writable temp / multi-tenant) can swap full_path for a symlink to e.g. /etc/shadow in the window after line 47, defeating the containment control (a static symlink would be resolved and rejected at check time, so the race is the only bypass). CWE-367 is the correct classification; the impact (arbitrary-file map-read whose bytes land in a weight tensor, exfiltratable via model output) is accurate but gated on the heavy precondition of attacker write access to the model dir plus winning a micro-window race, so this is a real-but-environment-dependent defect, not a memory-safety bug. The proposed fix is the correct direction: open once after sanitize and operate on the fd (fstat for size, mmap(fd,...)); O_NOFOLLOW additionally refuses a symlinked final component atomically. Caveat: O_NOFOLLOW only guards the last component, so the definitive fix is fd-based size+map throughout (and on Windows use a single CreateFile handle with GetFileSizeEx + CreateFileMapping). The 're-run weakly_canonical before each call' alternative only shrinks the window and should not be the chosen fix.

## Exploit / Proof of Concept
1. Craft an ONNX model whose external-data `location` names a file that does not yet exist in the model directory (e.g., `weights_toctou.bin`). 2. At check time `sanitize_path` calls `weakly_canonical(model_dir / 'weights_toctou.bin')`; because the file is absent, `weakly_canonical` returns `<canonical_model_dir>/weights_toctou.bin` with the filename appended unresolved — the containment check passes. 3. In the race window between `sanitize_path` returning (line 47) and `ov::load_mmap_object(full_path)` executing (line 62), the attacker creates a symlink `<model_dir>/weights_toctou.bin -> /etc/shadow`. 4. `file_size` and `load_mmap_object` follow the newly created symlink, bypassing the containment check and mapping `/etc/shadow` into the process. The same race exists on the `file_size` → `load_mmap_object` boundary even for pre-existing files: replace the file with a symlink between those two calls.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression intent for TOCTOU at tensor_external_data.cpp:47-62 / 74-113.
// A deterministic race is not testable in gtest; instead this skeleton encodes
// the containment invariant the fd-based fix must still satisfy: an external-data
// 'location' that resolves (via a symlink in model_dir) outside the base dir must
// be REJECTED, and once the fix opens with O_NOFOLLOW / fd-based I/O the symlinked
// final component must not be followed.
//
// Style mirrors src/frontends/onnx/tests/onnx_import.in.cpp.
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

// TODO: provide a crafted .onnx whose tensor external_data.location names
//       "weights_toctou.bin" (a file absent at parse time) under models/.
static std::string s_model = "external_data/external_data_toctou.onnx";

TEST(onnx_external_data_toctou, symlink_escapes_base_is_rejected) {
    // TODO: in the test fixture, create models/external_data/weights_toctou.bin
    //       as a symlink pointing outside the model directory (e.g. to a temp
    //       file standing in for /etc/shadow). std::filesystem::create_symlink.
    //       Skip on platforms/CI without symlink privilege.
    //
    // Pre-fix: sanitize_path passes (final component absent / appended verbatim),
    //          then file_size()+load_mmap_object() follow the symlink => leak.
    // Post-fix: open(O_NOFOLLOW)+fstat+mmap(fd) refuses the symlinked component,
    //          surfacing as ov::Exception / invalid_external_data.
    EXPECT_THROW(convert_model(s_model), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter=onnx_external_data_toctou.* . Pre-fix expectation: with a symlink at models/external_data/weights_toctou.bin pointing outside the base dir, the load follows the symlink (no throw / out-of-base mmap read); the EXPECT_THROW fails. Post-fix (fd-based open with O_NOFOLLOW): the symlinked final component is refused and convert_model throws ov::Exception, so the test passes. TODO: supply the crafted .onnx fixture and symlink setup; skeleton will not compile/run as-is.

## Suggested fix
Open the file with `open()` / `CreateFile()` immediately after the `sanitize_path` check and use the resulting file descriptor for all subsequent operations (size query via `fstat`, `mmap(fd, …)`), never re-resolving the path. On POSIX this means: `int fd = open(full_path.c_str(), O_RDONLY | O_NOFOLLOW); if (fd < 0) throw …; struct stat st; fstat(fd, &st);` then pass `fd` to `mmap`. `O_NOFOLLOW` additionally refuses to open a symlink target atomically. Alternatively, re-run `weakly_canonical` immediately before each open call and verify the result still satisfies the containment invariant, though this only shrinks the race window rather than eliminating it. The definitive fix is fd-based I/O throughout.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #394.
