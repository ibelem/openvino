# Security finding #124: m_offset is parsed from the untrusted ONNX protobuf 'offset' key by…

**Summary:** m_offset is parsed from the untrusted ONNX protobuf 'offset' key by…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** Arbitrary read of process memory: an attacker-crafted ONNX file can direct the OpenVINO ONNX frontend to read m_data_length bytes from any process-VA the attacker names as m_offset, and the resulting data is embedded into a Constant node. In a service that returns inference results or error messages this leaks heap/stack/code-segment content (ASLR bypass, credential leak). With a sufficiently large m_data_length it may also crash the process (access violation / SIGSEGV) — reliable DoS.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126` — `TensorExternalData::load_external_mem_data()`
**Validated for repos:** openvino
**Trust boundary:** ONNX model file on disk (or API-supplied bytes) → TensorProto::external_data 'offset' field → TensorExternalData constructor → load_external_mem_data()

## Description / Root cause
m_offset is parsed from the untrusted ONNX protobuf 'offset' key by std::stoull (constructor line 24) with no range or alignment constraint. At line 126 it is reinterpret_cast<char*>(m_offset) — a raw, attacker-controlled 64-bit integer cast to a char pointer. At line 129 std::memcpy reads m_data_length bytes from that address into an AlignedBuffer. The only guard (lines 121-123) rejects offset==0 with data_length>0; any non-zero offset (e.g. 1) bypasses it. Because ORT_MEM_ADDR ("*/_ORT_MEM_ADDR_/*") is a public compile-time constant in the header (tensor_external_data.hpp:91), any ONNX file can set location to that value and enter this code path.

**Validator analysis:** The flaw is real and reachable from ov::Core::read_model on the OpenVINO repo's trust boundary. For a model loaded off disk, Tensor::m_tensor_place is nullptr, so get_ov_constant() (tensor.cpp:449-453) constructs TensorExternalData directly from the untrusted TensorProto; has_external_data() (tensor.hpp:312-313) returns true for EXTERNAL location, and the location string is compared against the publicly-defined ORT_MEM_ADDR constant (hpp:91). The only guard in load_external_mem_data (line 121) merely rejects offset==0/length==0; any offset>=1 reaches line 126's reinterpret_cast<char*>(m_offset) and line 129's memcpy, giving an arbitrary process-memory read whose bytes are embedded in a Constant. Note this dangerous branch runs BEFORE the element_count/shape consistency check at tensor.cpp:467, so it executes for any shape. The same sink also exists in get_external_data() (tensor.hpp:324-325). CWE-822 (Untrusted Pointer Dereference) is the correct classification; the dual impact (info-leak via returned constant, and DoS/SIGSEGV with a large length) is accurate. The proposed fix's primary clause is correct and sufficient: gate the ORT_MEM_ADDR branch on m_tensor_place != nullptr (a non-null place is only produced by ORT's trusted in-process tensor-sharing path), so a TensorProto-derived tensor can never reach load_external_mem_data — it should then fall through to the file-based loaders which already bound-check offset against file_size (lines 53-54, 83-84). The fix must be applied at BOTH call sites (tensor.cpp:455 and tensor.hpp:324). The secondary suggestion (an alignment OPENVINO_ASSERT inside load_external_mem_data) is insufficient on its own — alignment does not prevent an arbitrary in-bounds-of-VA read; without a validated address range the cast remains exploitable, so the source-gating clause is the load-bearing part of the fix.

## Exploit / Proof of Concept
Craft an ONNX model containing an initializer tensor with external_data entries: {key:'location', value:'*/_ORT_MEM_ADDR_/*'}, {key:'offset', value:'<target VA as decimal, e.g. 4096>'}, {key:'length', value:'65536'}. Pass this model to ov::Core::read_model(). Tensor::get_ov_constant() (tensor.cpp:453-456) constructs TensorExternalData(*m_tensor_proto) from the raw proto, sees data_location==ORT_MEM_ADDR, calls load_external_mem_data(). The guard at lines 121-123 passes (offset=4096 != 0 && length=65536 != 0). Line 126 produces addr_ptr=0x1000, line 129 memcpy copies 64 KB from that address into AlignedBuffer — process memory is now inside a Constant node. No authentication, no privilege, only model-load access is needed.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-822 in
//   openvino/src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp:126-129
//   (load_external_mem_data: reinterpret_cast<char*>(m_offset) + memcpy)
// reached via core/tensor.cpp:455-456 when a disk-loaded TensorProto sets
//   external_data location == "*/_ORT_MEM_ADDR_/*" and a non-zero offset.
//
// The assertion encodes the fix: a model whose initializer names the
// ORT_MEM_ADDR marker with an attacker-controlled offset MUST be rejected
// (ov::Exception / invalid_external_data) at convert_model time, never
// dereferenced. Pre-fix this test crashes under ASan (wild memcpy from
// reinterpret_cast<char*>(0x1000)); post-fix it throws and the EXPECT_THROW
// passes.
//
// NOTE (fallback skeleton): triggering the path requires a crafted .onnx
// fixture that carries external_data {location="*/_ORT_MEM_ADDR_/*",
// offset="4096", length="65536"} on an initializer. That binary fixture does
// not yet exist in the test data tree, so the model name below is a TODO.
OPENVINO_TEST(${BACKEND_NAME}, onnx_ort_mem_addr_offset_rejected) {
    // TODO: add a crafted model under onnx/frontend/tests/models/, e.g.
    //   external_data_ort_mem_addr_arbitrary_offset.onnx
    // It must contain one initializer with data_location=EXTERNAL and
    // external_data entries:
    //   {key:"location", value:"*/_ORT_MEM_ADDR_/*"}
    //   {key:"offset",   value:"4096"}
    //   {key:"length",   value:"65536"}
    // Pre-fix: convert_model dereferences addr 0x1000 -> ASan SEGV.
    // Post-fix: the ORT_MEM_ADDR branch is refused for TensorProto-derived
    //           tensors (m_tensor_place == nullptr) and an ov::Exception is
    //           thrown.
    EXPECT_THROW(convert_model("external_data_ort_mem_addr_arbitrary_offset.onnx"),
                 ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (configure OpenVINO with -DENABLE_TESTS=ON and -DENABLE_SANITIZER=ON for ASan). Run: ov_onnx_frontend_tests --gtest_filter='*onnx_ort_mem_addr_offset_rejected*'. Expected pre-fix: AddressSanitizer SEGV / 'unknown-address' read inside std::memcpy at tensor_external_data.cpp:129 (deref of reinterpret_cast<char*>(0x1000)). Expected post-fix: test passes — convert_model throws ov::Exception (invalid_external_data) before any dereference. TODO before running: author the crafted .onnx fixture named in the test and place it in the onnx frontend tests models directory.

## Suggested fix
The ORT_MEM_ADDR code path must NEVER be reachable from a model loaded off disk or from an API-supplied byte buffer. Fix at the call site in Tensor::get_ov_constant() (tensor.cpp:455-456): refuse the ORT_MEM_ADDR branch when the tensor was constructed from a TensorProto (i.e. m_tensor_place == nullptr) — only allow it when m_tensor_place is non-null and was set by ORT's in-process tensor sharing path. Inside load_external_mem_data() itself, add a static assertion or OPENVINO_ASSERT that m_offset is aligned to at least sizeof(void*) and falls within an explicitly passed valid address range rather than trusting the raw integer.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #124.
