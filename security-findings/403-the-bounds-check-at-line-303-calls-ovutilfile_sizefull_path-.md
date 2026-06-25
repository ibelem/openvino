# Security finding #403: The bounds check at line 303 calls `ov::util::file_size(full_path)`…

**Summary:** The bounds check at line 303 calls `ov::util::file_size(full_path)`…

**CWE IDs:** CWE-908: Use of Uninitialized Resource / CWE-367: TOCTOU Race Condition
**Severity / Impact:** If the external data file is truncated between the `file_size()` call and the `read()` call (TOCTOU), the returned tensor buffer contains partially uninitialized heap data. These bytes are handed back as valid model weights, silently corrupting inference results. Under adversarial control the uninitialized bytes may contain sensitive heap residue (key material, prior model weights), constituting an information-disclosure path. The stream is also cached (`(*cache)[full_path] = external_data_stream`), so subsequent tensors read from the same file via the cached stream will inherit a stream that is in `failbit` / `eofbit` state, causing all future reads to silently return 0 bytes.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:359` — `extract_tensor_external_data()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-writable external data file referenced by an ONNX model; the filesystem path is fully attacker-controlled via the model's external_data[].location/offset/length fields

## Description / Root cause
The bounds check at line 303 calls `ov::util::file_size(full_path)` and validates `ext_data_offset`/`ext_data_length` against it, then at line 357 `seekg(ext_data_offset)` is called and at lines 359-360 `external_data_stream->read(data_ptr, resolved_data_length)` is issued — but neither `seekg`'s success (line 357) nor `read()`'s actual transfer count (no `gcount()` call, no `fail()` check after the read) is ever verified. The allocated buffer at line 353 can therefore contain uninitialized/stale heap bytes for any bytes not transferred.

**Validator analysis:** The code defect is real: the buffer is allocated with `new uint8_t[size]` (graph_iterator_proto.hpp:136) which is uninitialized, and the read at cpp:359-360 never verifies bytes transferred (no gcount(), no fail()/eof() check), nor does seekg at cpp:357 get checked. However the impact is materially overstated. The bounds check at cpp:304-305 validates ext_data_offset/ext_data_length against file_size() and throws on any short/oversized request, so in normal (non-racing) operation the requested bytes are always available and the read fully populates the buffer — there is NO uninitialized data on the static path. The only residual exposure is the genuine TOCTOU window (CWE-367) where an attacker who can already write the external-data file truncates it between file_size() at cpp:303 and read() at cpp:359, plus the secondary cache-poisoning of the cached ifstream (failbit/eofbit persisting in m_stream_cache). Both require winning a tight race on a file the attacker already controls, so practical severity is low-to-moderate, not the silent weight-corruption/heap-disclosure-on-every-load implied. CWE-908 is accurate for the race outcome; the 'silently corrupting all inference' framing is the worst case of a narrow race, not the default. The proposed fix is correct and sufficient: checking external_data_stream->fail()/gcount() after read and fail() after seekg closes both the uninitialized-buffer window and the stream-poisoning, converting the condition into a thrown runtime_error at the API boundary. Recommend also zero-initializing the allocation (value-init `new uint8_t[size]()` in allocate_data) as defense-in-depth so any future short read can never leak heap residue. A deterministic gtest cannot reproduce the short read (the bounds check guarantees a full read for any static fixture; only a concurrent truncation race produces it), hence a skeleton test is provided.

## Exploit / Proof of Concept
1. Supply an ONNX model referencing an external data file `weights.bin` with `offset=0` and `length=N`. 2. During model loading, after `file_size()` returns N at line 303 but before `read()` at line 359, truncate `weights.bin` to 0 bytes (race with a concurrent write or a symlink swap). 3. `seekg(0)` succeeds vacuously; `read(data_ptr, N)` transfers 0 bytes and sets `eofbit`/`failbit`, but `data_ptr` still holds N bytes of raw uninitialized heap. 4. `return true` propagates `tensor_meta_info.m_tensor_data = data_ptr` as if the weights were loaded correctly. Any use of `data_ptr` as tensor input now reads uninitialized/stale heap memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression target: openvino/src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp:357-360
// extract_tensor_external_data() reads external tensor data into an uninitialized
// `new uint8_t[size]` buffer without verifying seekg()/read() success or gcount().
// The fix adds: throw if seekg fails, and throw if read fails / gcount != expected size.
//
// What this test encodes:
//   Loading a model whose external_data file does NOT deliver the declared number
//   of bytes at read() time must be REJECTED (throw ov::Exception / runtime_error),
//   instead of silently returning a buffer with uninitialized heap residue.
//
// NOTE: This cannot be reproduced with an ordinary static fixture, because the
// bounds check at graph_iterator_proto.cpp:304-305 validates the declared
// offset/length against ov::util::file_size() BEFORE the read and throws on any
// mismatch. A genuine short read only occurs when the file is truncated between
// file_size() (cpp:303) and read() (cpp:359) — a TOCTOU race — or via a special
// file (FIFO/device) whose seekg/read short-transfers. Both require fixture
// machinery not expressible as a plain .onnx, so this is a SKELETON.

#include "onnx_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include <gtest/gtest.h>

using namespace ov::frontend::onnx::tests;

// TODO: place alongside the other external-data tests in onnx_import.in.cpp
// and register MANIFEST entry; target binary: ov_onnx_frontend_tests.
OPENVINO_TEST(${FRONTEND_NAME}_onnx_import, model_external_data_short_read_is_rejected) {
    // TODO: build a crafted model + external data file such that the external
    //       data stream delivers FEWER bytes than declared at read() time while
    //       still passing the file_size() bounds check at cpp:304-305.
    //       Options:
    //         (a) a FIFO/named pipe or /dev-style special file as external_data location
    //             whose file_size() reports the declared length but seekg/read short-transfer;
    //         (b) a test harness hook that truncates the backing file between
    //             ov::util::file_size() and ifstream::read() (TOCTOU).
    //       Neither is a static .onnx fixture, hence the TODO.
    //
    // Pre-fix expectation: convert_model() returns successfully and the weights
    //   tensor contains uninitialized heap bytes (catchable under MSan/ASan as a
    //   use-of-uninitialized-value when the constant is consumed).
    // Post-fix expectation: convert_model() throws because read() short-transfers
    //   (gcount() != declared size) or seekg() fails.
    EXPECT_THROW(convert_model("external_data/short_read_external_data.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build: cmake --build . --target ov_onnx_frontend_tests . Run: ./ov_onnx_frontend_tests --gtest_filter='*model_external_data_short_read_is_rejected*' . Build with -DENABLE_SANITIZER=ON (or MSan) so that, pre-fix, consuming the partially-read external weights surfaces a use-of-uninitialized-value (MSan) / heap-buffer read; post-fix the conversion throws ov::Exception and the test passes. TODO requires a FIFO/special-file or a truncation hook fixture (not a static .onnx) — see in-test TODOs.

## Suggested fix
After the `read()` call at line 360, add: `if (external_data_stream->fail() || static_cast<size_t>(external_data_stream->gcount()) != tensor_meta_info.m_tensor_data_size) { throw std::runtime_error("Short or failed read of external tensor data: expected " + std::to_string(tensor_meta_info.m_tensor_data_size) + " bytes, got " + std::to_string(external_data_stream->gcount())); }`. Also check `seekg` success: after line 357, add `if (external_data_stream->fail()) { throw std::runtime_error("seekg failed for external data stream"); }`. These two checks close both the TOCTOU uninitialized-data window and the silent stream-poisoning of the cache.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #403.
