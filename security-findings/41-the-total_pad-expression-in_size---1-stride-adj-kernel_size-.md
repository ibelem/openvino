# Security finding #41: The total_pad expression `(in_size - 1) * stride + adj + (kernel_si…

**Summary:** The total_pad expression `(in_size - 1) * stride + adj + (kernel_si…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Signed integer overflow is undefined behaviour in C++. Under optimising compilers the UB can propagate unpredictably: the resulting `total_pad` value is written into `pads_begin[i]` / `pads_end[i]` (CoordinateDiff / ptrdiff_t) at lines 223-224 and passed directly to `v1::ConvolutionBackpropData` or `v1::GroupConvolutionBackpropData` constructors. Downstream shape-inference code that trusts these pads to compute output sizes can then allocate buffers of grossly wrong size, triggering an out-of-bounds write or a heap allocation failure that crashes the process. Any application that loads user-supplied ONNX models is at risk.
**Affected location:** `targets/openvino/src/frontends/onnx/frontend/src/op/conv_transpose.cpp:218` — `conv_transpose()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted ONNX model file → ov::frontend::onnx ConvTranspose op handler

## Description / Root cause
The total_pad expression `(in_size - 1) * stride + adj + (kernel_size - 1) * dilation + 1 - out_size` at lines 218-219 operates entirely in signed int64_t with no overflow guard. `stride` is assigned from `strides[i]` (a `size_t` element of `ov::Strides`) at line 211 with no range check; an attacker-supplied stride value of e.g. `0x4000000000000001` (4611686018427387905, fits in both size_t and int64_t) combined with `in_size >= 3` makes `(in_size-1) * stride` exceed INT64_MAX, invoking signed-integer UB. Likewise `out_size` at line 214 comes directly from the ONNX `output_shape` int64 attribute; if the model supplies INT64_MIN, the subtraction `1 - out_size` also overflows. No saturation, assert, or range guard exists between reading these attribute values and performing the multiplication/subtraction.

**Validator analysis:** The defect is real. strides is read via convpool::get_strides -> get_attribute_value<vector<size_t>> (convpool.cpp:59-73) with NO range validation; a value like 0x4000000000000001 (2^62+1) fits in both size_t and int64_t. At conv_transpose.cpp:211 it is assigned to int64_t stride; at 218-219 (in_size-1)*stride is computed (in_size=3 -> 2*4611686018427387905 = 9223372036854775810 > INT64_MAX) BEFORE std::max clamps, producing signed-integer overflow which is UB (CWE-190). out_size from the int64 output_shape attribute (line 214) can likewise make 1-out_size overflow when out_size=INT64_MIN. The branch at line 203 is reachable from any untrusted ONNX model with non-empty output_shape and static data/filter shapes. CWE-190 is accurate. The IMPACT claim of a concrete OOB write is OVERSTATED: the wrapped-around total_pad flows into CoordinateDiff pads forwarded to v1::ConvolutionBackpropData, where shape inference will far more plausibly throw or produce a wrong shape than perform an attacker-controlled OOB write; the demonstrable consequence is signed-overflow UB and potential crash/garbage shape, not a proven memory-corruption primitive. The proposed fix is correct and sufficient in direction: add CHECK_VALID_NODE that strides[i]/dilations[i] are in a sane positive range, output_shape[i] > 0, and output_padding[i] < strides[i] per the ONNX spec, OR do the arithmetic in __int128 / a checked-multiply helper and throw on overflow. Bounding strides/dilations to a small architecture limit (e.g. [1, 65535]) is the simplest robust guard and also fixes the related calculate_transpose_auto_pads path (convpool.cpp:196,211) which has the same unbounded multiply.

## Exploit / Proof of Concept
Craft an ONNX ConvTranspose node whose `strides` INTS attribute contains a single element `4611686018427387905` (= 2^62 + 1). Provide a static input data shape [1, 1, 3] (so in_size = 3 → in_size-1 = 2). Provide a non-empty `output_shape` attribute so the overflow branch at line 203 is entered. At line 211, `stride = strides[0] = 4611686018427387905` as int64_t. At line 219, `(3-1) * 4611686018427387905 = 9223372036854775810 > INT64_MAX`; signed overflow fires. The corrupted total_pad is then distributed into pads_begin/pads_end and forwarded to ConvolutionBackpropData shape inference without any sanitisation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-190 signed-integer overflow in
//   openvino/src/frontends/onnx/frontend/src/op/conv_transpose.cpp:218-219
// Pre-fix: convert_model on a ConvTranspose with strides=[4611686018427387905]
//          and output_shape=[1] over a static [1,1,3] input triggers signed
//          overflow (UBSan: "signed integer overflow" in conv_transpose) or a
//          garbage pad fed to ConvolutionBackpropData.
// Post-fix: the frontend rejects the out-of-range stride and throws ov::Exception.
//
// Harness: ov_onnx_frontend_tests (gtest + ASan/UBSan), style of onnx_import.in.cpp.
// NOTE: this requires a crafted model fixture. The exact strides value cannot be
// expressed in the prototxt the auto-generator emits without a real serialized
// model, so a binary .onnx fixture is needed -> see TODOs below.

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov::frontend::onnx::tests;

static std::string s_manifest = "${MANIFEST}";

// TODO: place a crafted model at
//   openvino/src/frontends/onnx/tests/models/conv_transpose_stride_overflow.onnx
// containing a single ConvTranspose node with:
//   input  X : tensor<float>[1,1,3]
//   input  W : tensor<float>[1,1,1]
//   attr   strides    = [4611686018427387905]   (2^62 + 1)
//   attr   output_shape = [1]
// (output_shape must be non-empty and shapes static so the line-203 branch runs).
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv_transpose_stride_overflow) {
    // Pre-fix: signed-overflow UB at conv_transpose.cpp:218-219 (UBSan trap) or a
    // bogus graph. Post-fix: CHECK_VALID_NODE rejects the out-of-range stride.
    EXPECT_THROW(convert_model("conv_transpose_stride_overflow.onnx"), ov::Exception);
}
```
**Build / run:** Build target: ov_onnx_frontend_tests. Run: ov_onnx_frontend_tests --gtest_filter='*onnx_model_conv_transpose_stride_overflow*'. Build with -fsanitize=undefined (and address). Expected pre-fix: UBSan reports 'signed integer overflow: ... * ... cannot be represented in type long' at frontends/onnx/frontend/src/op/conv_transpose.cpp:219 (or a crash in BackpropData shape inference); post-fix the test passes because convert_model throws ov::Exception from CHECK_VALID_NODE. TODO: add the crafted conv_transpose_stride_overflow.onnx fixture under frontends/onnx/tests/models/ and register it in the manifest.

## Suggested fix
Before the total_pad calculation, validate each of stride, dilation, adj, and out_size against safe ranges. Specifically: (1) CHECK_VALID_NODE that `strides[i]` and `dilations[i]` are in [1, 65536] (or a reasonable architecture limit); (2) CHECK_VALID_NODE that `output_shape[i] > 0`; (3) CHECK_VALID_NODE that `output_padding[i] < strides[i]` per ONNX spec. Alternatively, perform the arithmetic in `__int128` or use a checked-multiply helper and throw CHECK_VALID_NODE on overflow before assigning to total_pad.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #41.
