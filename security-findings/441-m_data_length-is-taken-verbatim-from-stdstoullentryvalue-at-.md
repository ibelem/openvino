# Security finding #441: `m_data_length` is taken verbatim from `std::stoull(entry.value())`…

**Summary:** `m_data_length` is taken verbatim from `std::stoull(entry.value())`…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** A crafted ONNX model with `location="*/_ORT_MEM_ADDR_/*"` and `length=0xFFFFFFFFFFFF0000` triggers an attempt to allocate ~16 exabytes. If `util::aligned_alloc` throws `std::bad_alloc`, it propagates past `get_ov_constant`'s catch (which only catches `ov::Exception`, tensor.cpp:481) and terminates the process — denial of service. If the allocator returns `nullptr` instead of throwing (valid on some platforms/configurations), `m_aligned_buffer` is set to null, and the subsequent `memcpy(nullptr, ..., m_data_length)` at line 129 causes a segfault — also DoS. Single-request, no authentication required.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:127` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file (attacker-supplied) → AlignedBuffer allocator via ORT_MEM_ADDR path

## Description / Root cause
`m_data_length` is taken verbatim from `std::stoull(entry.value())` at line 26 with no upper-bound check, then passed directly to `ov::AlignedBuffer(m_data_length)` at line 127 inside `load_external_mem_data()`. Unlike the file-based paths (`load_external_data` line 83–85, `load_external_mmap_data` line 53–56) which both guard against exceeding actual file size, this path has only a zero/nonzero check (line 121–124) and no size cap whatsoever.

**Validator analysis:** The defect is real for the standalone OpenVINO ONNX frontend: TensorExternalData reads m_data_length verbatim from std::stoull (line 26) and load_external_mem_data() (line 116-134) allocates ov::AlignedBuffer(m_data_length) (line 127) guarded only by a nonzero check (line 121-124), whereas load_external_data (83-85) and load_external_mmap_data (53-56) cap m_data_length against the real file size. AlignedBuffer's ctor (aligned_buffer.cpp:18-19) does util::aligned_alloc(m_byte_size) with no validation, so a ~16EB request throws std::bad_alloc; the call site Tensor::get_ov_constant invokes the loader at tensor.cpp:456 OUTSIDE the try/catch (479-485, which catches only ov::Exception around Constant construction), so bad_alloc escapes that handler. The element_count!=shape_elements check (tensor.cpp:467) happens AFTER the allocation, so it cannot prevent it. CWE-789 (excessive-size allocation) is the correct category; impact is DoS (bad_alloc propagation, or, if alloc returns null on some configs, null-deref at memcpy line 129). Note the separate, arguably worse issue that m_offset is treated as a raw address and memcpy'd from (line 129) — an arbitrary-read — which the finding mentions but does not center; that is not fixed by a size cap alone. The proposed fix (explicit upper-bound check before AlignedBuffer, plus null-check in the ctor and a bad_alloc try/catch in get_ov_constant) is correct and sufficient for the CWE-789 claim; a tighter fix would derive the cap from element_count*element_size/shape consistency like the other two paths and additionally validate the m_offset address provenance. For OVEP, the same code is the sink but the length is constrained by ORT's prior allocation, so it is not reachable as an unbounded value from that boundary.

## Exploit / Proof of Concept
Craft an ONNX model with an initializer whose `external_data` has `key=location, value=*/_ORT_MEM_ADDR_/*` and `key=length, value=18446744073709486080` (0xFFFFFFFFFFFF0000). When the OV EP loads this model via `Tensor::get_ov_constant`, it calls `load_external_mem_data()`. The check at line 121–124 passes (m_data_length != 0 and m_offset != 0). `AlignedBuffer(0xFFFFFFFFFFFF0000)` is constructed — the allocator either throws `std::bad_alloc` (process crash) or returns null (null-deref crash at memcpy).

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-789 at
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:121-127
// (load_external_mem_data allocates AlignedBuffer(m_data_length) with no upper bound).
// Pre-fix: a crafted ONNX model with external_data location="*/_ORT_MEM_ADDR_/*" and
//          length=0xFFFFFFFFFFFF0000 reaches Tensor::get_ov_constant -> load_external_mem_data
//          and triggers a ~16EB allocation -> std::bad_alloc escapes the ov::Exception catch
//          at tensor.cpp:479-485 (test process aborts / ASan reports allocation-size-too-big).
// Post-fix: the added size cap throws ov::frontend::onnx::error::invalid_external_data
//          (an ov::Exception), so convert_model rejects the model cleanly.
//
// Style follows src/frontends/onnx/tests/onnx_import.in.cpp (uses convert_model helper
// and the manifest-driven test fixture). A crafted .onnx fixture is required because the
// ORT_MEM_ADDR external_data fields cannot be expressed without a serialized model.
TEST(${BACKEND_NAME}, onnx_external_mem_data_excessive_length_is_rejected) {
    // TODO: provide crafted fixture at
    //   src/frontends/onnx/tests/models/external_data/ort_mem_addr_excessive_length.onnx
    //   containing one initializer with data_location=EXTERNAL and external_data entries:
    //     key="location", value="*/_ORT_MEM_ADDR_/*"
    //     key="offset",   value="<any nonzero, e.g. an arbitrary placeholder address>"
    //     key="length",   value="18446744073709486080"   // 0xFFFFFFFFFFFF0000
    // NOTE: offset is interpreted as a raw pointer; keep length the trigger so the
    //       allocation aborts BEFORE the memcpy at line 129 is reached.
    EXPECT_THROW(convert_model("external_data/ort_mem_addr_excessive_length.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (build with -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_external_mem_data_excessive_length_is_rejected*'. Expected pre-fix: process aborts via uncaught std::bad_alloc, or ASan 'requested allocation size 0xfffffffffff... exceeds maximum supported size' at AlignedBuffer ctor (aligned_buffer.cpp:19) reached from tensor_external_data.cpp:127. Expected post-fix: test passes because load_external_mem_data throws error::invalid_external_data (ov::Exception). TODO: add the crafted .onnx fixture and register it in the models manifest before this compiles/runs.

## Suggested fix
Add an explicit upper-bound check in `load_external_mem_data()` before constructing `AlignedBuffer`. For example, after line 125 insert: `if (m_data_length > (1ULL << 32)) { throw error::invalid_external_data{*this}; }` (adjust limit to a reasonable maximum tensor size). Additionally, check the result of `util::aligned_alloc` in the `AlignedBuffer` constructor and throw `std::bad_alloc` if it returns null, and add `try { ... } catch (const std::bad_alloc&) { throw error::invalid_external_data{...}; }` around the `AlignedBuffer` construction in `get_ov_constant`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #441.
