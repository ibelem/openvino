# Security finding #129: At line 120 `axis = slice_rule.axis` is an unchecked `int`. It is t…

**Summary:** At line 120 `axis = slice_rule.axis` is an unchecked `int`. It is t…

**CWE IDs:** CWE-129: Improper Validation of Array Index
**Severity / Impact:** OOB read (line 129) and OOB write (lines 131, 136, 137, 145) on attacker-controlled index into heap vectors and DNNL descriptor arrays. Depending on what is overwritten this is at minimum an exploitable crash/DoS; on a sufficiently controlled write primitive (e.g., negative axis converting to a large size_t offset, writing `abs_stride` over adjacent heap metadata) this could enable arbitrary code execution in the inference process.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129` — `PortIteratorHelper::PortIteratorHelper()`
**Validated for repos:** openvino
**Trust boundary:** PortMap.axis is populated at lines 520 and 548 directly from model-graph attributes (ConcatOutputDescription::m_axis / SliceInputDescription::m_axis) in TensorIterator::createPrimitive, without bounds validation before it reaches the constructor.

## Description / Root cause
At line 120 `axis = slice_rule.axis` is an unchecked `int`. It is then used at line 129 as an index into `full_dims` (a `std::vector<size_t>` of rank-length), at line 131 to write into that vector, at line 136/137 to index into `chunk_desc.get()->dims[]` (a fixed-size raw C array of DNNL_MAX_NDIMS elements), and at line 145 into `format_desc.blocking.strides[axis]`. The only pre-filter (lines 764 and 780) rejects only the sentinel value `-1`; negative values such as -2 or any value >= `full_dims.size()` are silently passed to `PortIteratorHelper`.

**Validator analysis:** CWE-129 is accurate for the openvino CPU plugin. PortIteratorHelper at tensoriterator.cpp:120-145 uses PortMap.axis as a raw int index with no bounds/normalization. The researcher's headline example (axis=100 on a static rank-4 input) is actually MITIGATED: core op v0::TensorIterator::validate_and_infer_types (tensor_iterator.cpp:85-91) writes out_shape[axis]=part_size, and PartialShape::operator[] → normalize_shape_index (shape_util.cpp:63-70) throws OPENVINO_THROW for positive-out-of-range when rank is static. However the defect is still real and reachable two ways: (1) a negative-but-in-range axis such as -2, which the core happily normalizes to a valid dim (so the model loads), but the CPU plugin uses the raw -2 as a std::vector/raw-array index → size_t(-2) far-OOB read at :129 and OOB writes at :131/:136/:145; (2) a dynamic-rank input where line 86's static-rank branch is skipped entirely, leaving axis unchecked when concrete static dims arrive at createPrimitive. Impact (OOB read + OOB write of abs_stride into heap vectors / DNNL descriptor arrays) is plausible; 'arbitrary code execution' is speculative — DoS/memory-corruption is the defensible claim. The proposed fix (assert 0<=axis<full_dims.size() and axis<DNNL_MAX_NDIMS) stops the OOB but is INCOMPLETE/INCORRECT because it would reject legitimate negative axes that OpenVINO semantics permit. Better fix: normalize the axis against the known input/output rank at createPrimitive (tensoriterator.cpp:518-551) using ov::util::normalize/normalize_axis, then assert the normalized value is within [0,rank) before storing into PortMap, so valid negative axes still work and only truly out-of-range values are rejected.

## Exploit / Proof of Concept
Craft an ONNX or OpenVINO IR model containing a TensorIterator or Loop op whose SliceInputDescription has m_axis set to a value outside [0, input_rank) — for example axis=100 on a 4-D input tensor (full_dims.size()==4). When OpenVINO loads and runs inference on this model, TensorIterator::createPrimitive stores axis=100 directly into inputPortMap, prepareInputPorts() at line 768 calls new PortIteratorHelper(…, map_rule, …), and line 129 executes `full_dims[100]` on a 4-element vector — out-of-bounds heap read. Lines 131, 136, and 145 then perform out-of-bounds writes. A negative value such as -2 (which passes the axis!=-1 guard) promotes to a huge size_t, causing a far-OOB access.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-129 at openvino/src/plugins/intel_cpu/src/nodes/tensoriterator.cpp:129
// (PortIteratorHelper uses PortMap.axis as an unchecked index into full_dims / dnnl dims[]).
// Pre-fix: a TensorIterator with a SliceInputDescription whose m_axis is negative-in-range
// (e.g. -2 on a rank-4 input) loads fine in core (PartialShape::operator[] normalizes it) but
// the CPU plugin uses the raw -2 as a std::vector index -> size_t(-2) OOB (ASan heap-buffer-overflow).
// Post-fix: createPrimitive normalizes/validates the axis and the graph either runs correctly or
// throws ov::Exception instead of corrupting memory.
//
// TODO: This is a SKELETON. Building a TensorIterator body graph + SliceInputDescription with a
//       crafted m_axis requires the intel_cpu unit harness (ov_cpu_unit_tests). Fill in:
//       - construction of the inner body ov::Model (Parameter -> some op -> Result),
//       - ov::op::v0::TensorIterator with set_sliced_input(param, data, start, stride, part, end, AXIS),
//         where AXIS is the malicious value (-2) [confirm exact set_sliced_input signature in
//         openvino/src/core/include/openvino/op/tensor_iterator.hpp],
//       - compile_model on CPU and infer to force createPrimitive()/PortIteratorHelper.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(TensorIteratorAxisBounds, NegativeInRangeSliceAxisIsRejectedOrNormalized) {
    using namespace ov;
    // TODO: build inner body model
    // auto body_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1,1,1,1});
    // auto body_res   = std::make_shared<op::v0::Result>(body_param);
    // auto body = std::make_shared<Model>(ResultVector{body_res}, ParameterVector{body_param});

    // auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4,1,1,1});
    // auto ti = std::make_shared<op::v0::TensorIterator>();
    // ti->set_body(body);
    // const int64_t MALICIOUS_AXIS = -2; // legal-after-normalization for core, raw OOB in CPU plugin
    // ti->set_sliced_input(body_param, data, 0, 1, 1, -1, MALICIOUS_AXIS);
    // ti->get_concatenated_slices(body_res, 0, 1, 1, -1, MALICIOUS_AXIS);
    // auto model = std::make_shared<Model>(ti->outputs(), ParameterVector{data});

    // Core core;
    // auto compiled = core.compile_model(model, "CPU");
    // auto req = compiled.create_infer_request();
    // req.set_input_tensor(Tensor(element::f32, Shape{4,1,1,1}));
    // Pre-fix: OOB inside PortIteratorHelper (ASan). Post-fix: clean run or controlled throw.
    // EXPECT_NO_FATAL_FAILURE(req.infer());
    GTEST_SKIP() << "TODO: complete TensorIterator construction per intel_cpu unit harness";
}
```
**Build / run:** Build target: ov_cpu_unit_tests (cmake --build . --target ov_cpu_unit_tests). Run: ov_cpu_unit_tests --gtest_filter=TensorIteratorAxisBounds.* with -DENABLE_SANITIZER=ON. Expected pre-fix: AddressSanitizer 'heap-buffer-overflow READ/WRITE' inside PortIteratorHelper::PortIteratorHelper (tensoriterator.cpp:129/131). Expected post-fix: no ASan report — axis is normalized/validated in createPrimitive and infer() either succeeds or throws ov::Exception.

## Suggested fix
Insert an explicit bounds check immediately after line 120 before any use of `axis`: `OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < full_dims.size(), "TensorIterator: axis out of range for tensor rank");` and similarly assert `static_cast<size_t>(axis) < DNNL_MAX_NDIMS` before the raw dnnl descriptor accesses at lines 136–137 and 145. The same guard should be applied in TensorIterator::createPrimitive when populating inputPortMap/outputPortMap (lines 518–551), validating m_axis against the known input/output rank at that point.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #129.
