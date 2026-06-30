# Security finding #533: At tensor.hpp:197 the constructor calls `get_data_size()` inside `i…

**Summary:** At tensor.hpp:197 the constructor calls `get_data_size()` inside `i…

**CWE IDs:** CWE-248: Uncaught Exception (chained with CWE-20: Improper Input Validation)
**Severity / Impact:** Any attacker who can supply an ONNX model file (the common trust boundary for this frontend) can trigger an unhandled `ov::Exception` from inside the `Graph` constructor. At minimum this aborts model loading (functional DoS). If the calling application has no outer catch for `ov::Exception`/`std::exception`, the process terminates. Because ONNX initializers are processed unconditionally during graph construction, a single malformed initializer suffices.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/core/tensor.hpp:197` — `Tensor::Tensor(const TensorProto&, const std::filesystem::path&, detail::MappedMemoryHandles)()`
**Validated for repos:** openvino
**Trust boundary:** Attacker-controlled TensorProto parsed from an untrusted ONNX model file, arriving as an initializer in Graph::Graph()

## Description / Root cause
At tensor.hpp:197 the constructor calls `get_data_size()` inside `if (m_shape == ov::Shape{0} && get_data_size() == 1)` with no prior validation of `data_type()`. `get_data_size()` at line 375-376 checks `has_raw_data()` first and, if true, immediately calls `get_onnx_data_size(m_tensor_proto->data_type())` (utils.cpp:56) which has no case for `UNDEFINED` (value 0) and unconditionally throws `OPENVINO_THROW("unsupported element type")`. This throw escapes the Tensor constructor. The call site in graph.cpp:118 (`Tensor tensor = Tensor{initializer_tensor, ...}`) is OUTSIDE the try/catch block (graph.cpp:121-128) that is meant to absorb tensor conversion errors — that block only wraps `tensor.get_ov_constant()` at line 122, so the constructor-level exception is never caught there and propagates out of `Graph::Graph()`.

**Validator analysis:** The vuln type (CWE-248 Uncaught Exception / CWE-20 Improper Input Validation) is accurate: the Tensor constructor throws an ov::Exception for UNDEFINED data_type with raw_data set, and the call site at graph.cpp:118 is structurally outside the try/catch that was intended to absorb tensor errors (lines 121-128 only wrap get_ov_constant()). The impact (DoS / process abort) is accurate for applications without an outer catch. The proposed fix is correct in principle: (1) wrapping the Tensor{} constructor call at graph.cpp:118 inside the existing try/catch with a 'continue' to skip the bad initializer, and (2) adding an explicit UNDEFINED guard in get_data_size() or get_onnx_data_size(). However the proposed fix has a subtle incompleteness: if the constructor-level catch tries to call tensor.get_ov_type() for the failsafe constant (graph.cpp:127), that function ALSO throws for UNDEFINED (tensor.hpp:292). The safer fix is to use 'continue' to skip the initializer entirely on any exception from the constructor, not fall through to make_failsafe_constant().

## Exploit / Proof of Concept
Craft a binary ONNX protobuf with one initializer TensorProto: set `data_type=0` (UNDEFINED), `dims=[0]` (single dim of size 0, so OpenVINO maps it to `ov::Shape{0}`), and `raw_data` to any non-empty byte string (e.g. a single `0x00` byte). Feed this as the model file. Path: `Graph::Graph()` (graph.cpp:116) → `Tensor{initializer_tensor,...}` (graph.cpp:118, outside try/catch) → constructor:197 evaluates `m_shape == ov::Shape{0}` = TRUE → calls `get_data_size()` → tensor.hpp:375 `has_raw_data()` = TRUE → line 376 calls `get_onnx_data_size(0)` → utils.cpp:56 `OPENVINO_THROW("unsupported element type")` → exception escapes graph.cpp:118 bypassing the catch at graph.cpp:126.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for: tensor.hpp:197 / graph.cpp:118
// Flaw: Tensor constructor (called outside try/catch at graph.cpp:118) calls
//       get_data_size() which calls get_onnx_data_size(data_type=UNDEFINED=0)
//       at tensor.hpp:375-376, triggering an uncaught ov::Exception.
// Pre-fix: Graph::Graph() propagates the exception → import_onnx_model() throws.
// Post-fix: constructor call is inside try/catch with 'continue'; loading succeeds.
//
// Build target: ov_onnx_frontend_tests
// Filter:       --gtest_filter=OnnxImportRegression.UndefinedDtypeRawDataInitializer
// Expected sanitizer behavior (pre-fix): ov::Exception propagates uncaught from
//   Graph::Graph(), causing std::terminate or unhandled exception at call site.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "onnx/onnx_pb.h"
#include "onnx_import/onnx_import.hpp"
#include "openvino/frontend/exception.hpp"

namespace {
std::filesystem::path make_temp_model(const std::string& name) {
    // Build a minimal ONNX ModelProto with one initializer:
    //   data_type = 0 (UNDEFINED), dims = [0], raw_data = "\x00" (1 byte)
    // The combination dims=[0] => m_shape==ov::Shape{0} triggers the scalar
    // heuristic at tensor.hpp:197 which evaluates get_data_size(), reaching
    // the raw-data branch at tensor.hpp:375-376 and calling
    // get_onnx_data_size(UNDEFINED) which throws.
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(17);

    auto* graph = model.mutable_graph();
    graph->set_name("regression_533");

    // Malformed initializer
    auto* init = graph->add_initializer();
    init->set_name("bad_init");
    init->set_data_type(0);       // UNDEFINED
    init->add_dims(0);            // dims=[0] => ov::Shape{0}
    init->set_raw_data("\x00", 1); // has_raw_data() == true

    // Minimal graph output referencing the initializer so parsing proceeds
    auto* out = graph->add_output();
    out->set_name("bad_init");
    auto* ttype = out->mutable_type()->mutable_tensor_type();
    ttype->set_elem_type(1);  // FLOAT (placeholder)
    ttype->mutable_shape();   // empty shape

    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream ofs(path, std::ios::binary);
    EXPECT_TRUE(model.SerializeToOstream(&ofs));
    ofs.close();
    return path;
}
}  // namespace

TEST(OnnxImportRegression, UndefinedDtypeRawDataInitializer) {
    // Pre-fix:  this throws an ov::Exception escaping Graph::Graph().
    // Post-fix: the constructor call is inside the try/catch with 'continue';
    //           the bad initializer is skipped and loading succeeds (EXPECT_NO_THROW).
    const auto model_path = make_temp_model("regression_533_undefined_dtype.onnx");

    // After the fix, model loading must not throw.
    // (If the test fails with ov::Exception, the fix has not been applied.)
    EXPECT_NO_THROW({
        ov::frontend::onnx::import_onnx_model(model_path.string());
    });

    std::filesystem::remove(model_path);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests (with -DENABLE_SANITIZERS=ON for ASan).
Filter: --gtest_filter=OnnxImportRegression.UndefinedDtypeRawDataInitializer
Pre-fix expected failure: ov::Exception (message: 'unsupported element type' from get_onnx_data_size) propagates out of Graph::Graph(), causing the test to fail with an unexpected exception.
Post-fix expected result: PASS (EXPECT_NO_THROW succeeds; bad initializer is skipped by the widened try/catch with 'continue' at graph.cpp:118).

## Suggested fix
Two complementary fixes: (1) In `get_data_size()` (tensor.hpp:374-376), add a guard before the raw-data division: if `m_tensor_proto->data_type() == TensorProto_DataType_UNDEFINED` throw a descriptive `FRONT_END_THROW` (or return 0) rather than letting `get_onnx_data_size` throw anonymously. (2) In `Graph::Graph()` (graph.cpp:117-118), wrap the Tensor constructor call in the same try/catch that already handles `tensor.get_ov_constant()`, so that a malformed initializer is skipped rather than aborting the entire graph construction: `try { Tensor tensor = Tensor{...}; ... tensor.get_ov_constant(); ... } catch (const ov::Exception&) { /* skip malformed initializer */ continue; }`. Additionally, `get_onnx_data_size` in utils.cpp:55-56 should add `case TensorProto_DataType_UNDEFINED: FRONT_END_THROW("data_type UNDEFINED is not a valid tensor element type");` so the error is explicit.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #533.
