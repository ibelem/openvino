# Security finding #535: The Tensor constructor at tensor.hpp:197 unconditionally calls `get…

**Summary:** The Tensor constructor at tensor.hpp:197 unconditionally calls `get…

**CWE IDs:** CWE-369: Divide By Zero
**Severity / Impact:** Denial of service (crash) during ONNX model loading; any application parsing an untrusted ONNX model is affected. An attacker who can supply a crafted .onnx file can reliably crash the inference engine process on any call to `ov::Core::compile_model()` or equivalent.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:197` — `Tensor::Tensor(const TensorProto&, ...)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled TensorProto with data_type=UNDEFINED (0) and raw_data set, fed as a model initializer across the ONNX file-load trust boundary

## Description / Root cause
The Tensor constructor at tensor.hpp:197 unconditionally calls `get_data_size()` when `m_shape == ov::Shape{0}`, before any data_type validation. Inside `get_data_size()` (tensor.hpp:375-376), when `has_raw_data()` is true, the code computes `m_tensor_proto->raw_data().size() / get_onnx_data_size(m_tensor_proto->data_type())` with no guard that `data_type() != UNDEFINED`. When data_type is UNDEFINED (0), `get_onnx_data_size(0)` either returns 0 (integer divide-by-zero, UB → crash) or throws an unguarded exception. In contrast, `get_ov_constant()` (tensor.cpp:436) correctly calls `get_ov_type()` first, which throws for UNDEFINED — but the constructor does not have this protection.

**Validator analysis:** The vuln type (CWE-369 divide-by-zero / uncaught exception) and impact (DoS during model load) are accurate. The path is: attacker supplies TensorProto with dims:[0] (→ m_shape==ov::Shape{0}), data_type:UNDEFINED (0), non-empty raw_data → graph.cpp:118 constructs Tensor outside try/catch → tensor.hpp:197 condition fires → tensor.hpp:375-376 executes the division get_onnx_data_size(0). If get_onnx_data_size returns 0 for UNDEFINED, integer divide-by-zero UB occurs; if it throws, the exception propagates from the constructor at graph.cpp:118 which is outside the try/catch starting at line 121 that only guards get_ov_constant(). Either outcome is a DoS. The proposed fix is correct: add a data_type != UNDEFINED guard before calling get_data_size() in the constructor (tensor.hpp:197), or add an explicit check inside get_onnx_data_size() for the UNDEFINED case. A more robust fix would combine both: validate data_type at the top of the Tensor constructor and again inside get_onnx_data_size() to return 0 with a FRONT_END_THROW.

## Exploit / Proof of Concept
Craft an ONNX TensorProto initializer with: `dims: [0]`, `data_type: 0` (UNDEFINED), and a non-empty `raw_data` field. When the ONNX frontend processes it (graph.cpp:118 — `Tensor tensor = Tensor{initializer_tensor, ...}` — which is OUTSIDE the try/catch block that starts at line 121), the constructor fires, `m_shape == ov::Shape{0}` is true, `get_data_size()` is called, and the division `raw_data.size() / get_onnx_data_size(0)` triggers divide-by-zero or throws an uncaught exception, crashing the process.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-369 in Tensor::Tensor (tensor.hpp:197) + get_data_size() (tensor.hpp:375-376).
// A TensorProto with dims:[0], data_type:UNDEFINED (0), non-empty raw_data triggers divide-by-zero
// or uncaught exception when the Tensor constructor fires outside the try/catch at graph.cpp:118.
// Pre-fix: process crashes (SIGFPE from integer divide-by-zero) or throws unhandled exception.
// Post-fix: loading the model throws a controlled ov::Exception (rejected cleanly).

#include <gtest/gtest.h>
#include <fstream>
#include <string>

// ONNX protobuf headers (available in the ov_onnx_frontend_tests build environment)
#include <onnx/onnx_pb.h>

#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"

// Helper: write a minimal ONNX ModelProto to a temp file and return the path.
static std::string write_crafted_onnx(const std::string& path) {
    using namespace ONNX_NAMESPACE;

    ModelProto model;
    model.set_ir_version(7);
    model.set_opset_import(0)->set_version(17);
    model.mutable_opset_import(0)->set_domain("");

    GraphProto* graph = model.mutable_graph();
    graph->set_name("test_graph");

    // Craft the malicious initializer:
    //   dims: [0]         -> m_shape == ov::Shape{0}
    //   data_type: 0      -> UNDEFINED
    //   raw_data: 4 bytes -> non-empty, triggers the division
    TensorProto* init = graph->add_initializer();
    init->set_name("bad_tensor");
    init->add_dims(0);                      // dims: [0]
    init->set_data_type(0);                 // UNDEFINED
    init->set_raw_data("\xDE\xAD\xBE\xEF"); // non-empty raw_data

    // Add a trivial output to keep the graph structurally valid
    // (no nodes; the crash should happen at initializer parse time)
    ValueInfoProto* output = graph->add_output();
    output->set_name("out");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(1); // FLOAT

    std::ofstream ofs(path, std::ios::binary);
    model.SerializeToOstream(&ofs);
    return path;
}

TEST(OnnxImportUndefinedDataType, TensorCtorDivideByZeroUndefinedType) {
    // TODO: adjust tmp_path to a writable scratch directory available in the test environment.
    const std::string tmp_path = "/tmp/crafted_undefined_dtype.onnx";
    write_crafted_onnx(tmp_path);

    ov::Core core;
    // Pre-fix: this call crashes (SIGFPE) or throws std::terminate due to divide-by-zero in
    //   Tensor::get_data_size() at tensor.hpp:375-376 when data_type==UNDEFINED.
    // Post-fix: a controlled ov::Exception is thrown before the division.
    EXPECT_THROW(core.read_model(tmp_path), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (with -fsanitize=address,undefined).
Filter: --gtest_filter=OnnxImportUndefinedDataType.TensorCtorDivideByZeroUndefinedType
Expected pre-fix sanitizer error: UBSan 'divide-by-zero' in Tensor::get_data_size() at tensor.hpp:376, or SIGFPE crash;
Expected post-fix: test passes (exception caught cleanly by EXPECT_THROW).

## Suggested fix
Add an explicit guard in the Tensor constructor before calling `get_data_size()`: check `m_tensor_proto->data_type() != TensorProto_DataType_UNDEFINED` (or use `has_data_type()` and validate the type is known). Alternatively, wrap `get_onnx_data_size()` calls inside `get_data_size()` to throw or return a sentinel for UNDEFINED, and add a check at tensor.hpp:375: `if (m_tensor_proto->data_type() == TensorProto_DataType_UNDEFINED) OPENVINO_THROW("Cannot compute size of tensor with UNDEFINED data type");` before the division.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #535.
