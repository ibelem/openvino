# Security finding #378: At line 58, `in_offset = step * size` is computed using the element…

**Summary:** At line 58, `in_offset = step * size` is computed using the element…

**CWE IDs:** CWE-125: Out-of-bounds Read
**Severity / Impact:** For any i4 or u4 Concat with steps > 1 (concatenation_axis > 0), the reference kernel reads memory beyond the source tensor allocation for all steps after step 0. This is an attacker-triggerable OOB heap read that can crash the process (DoS) or leak adjacent heap contents (CWE-200 information disclosure) — affecting any application that runs inference on a model with an i4/u4 Concat op with a non-zero axis.
**Affected location:** `targets/openvino/src/core/reference/src/op/concat.cpp:58` — `concat()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Untrusted model/tensor data enters via TensorVector& inputs in Concat::evaluate() (concat.cpp:52), which calls reference::concat() with elem_size=elem_type.size()=1 for i4/u4

## Description / Root cause
At line 58, `in_offset = step * size` is computed using the element count (not yet halved). At line 60, `size /= 2` converts to byte count — but `in_offset` is never halved. In `copy_elements` (line 27), the source read is `arg + (in_offset * elem_size)` = `arg + (step * element_count_per_step * 1)`. For a packed i4/u4 buffer with N elements occupying N/2 bytes, the correct byte offset for step k is `k * (N/steps)/2`, but the code uses `k * (N/steps)` — a factor of 2 too large for every step > 0.

**Validator analysis:** The flaw is real. In reference::concat, for nibble types elem_size=elem_type.size()=1 and the packed buffer holds N elements in N/2 bytes. Line 58 sets in_offset=step*size where size=shape_sizes[i]/steps is still an ELEMENT count; line 60 then halves size to a byte count for the read length, but in_offset is never halved. In copy_elements the byte read is arg+(in_offset*1) with length (size*1), i.e. bytes [step*(N/steps), step*(N/steps)+(N/steps)/2). For steps>1 (any concatenation_axis with a non-unit product of leading dims) the max read at step=steps-1 reaches ~3N/4 bytes into an N/2-byte buffer → CWE-125 OOB heap read (DoS / CWE-200 info-leak). CWE-125 and the impact are accurate. Note: the exploit's concrete numbers are slightly wrong (two [1,4] i4 inputs on axis=1 give steps=out_shape[0]=1, NOT 2, so they do not trigger); a correct trigger is two i4 inputs shaped [2,4] concatenated on axis=1 (out=[2,8], steps=2): at step 1 in_offset=4 bytes into a 4-byte buffer, reading bytes 4..5 OOB. The proposed fix is correct and sufficient: divide size by 2 BEFORE computing in_offset for u4/i4 so in_offset becomes a true byte offset consistent with elem_size=1; equivalently halve in_offset for nibble types. Non-nibble types are unaffected because elem_size carries the real byte size and in_offset stays an element count.

## Exploit / Proof of Concept
Load an ONNX/IR model with a Concat node whose inputs have element type i4 or u4 and whose concatenation_axis > 0, so that `steps = shape_size(out_shape[0..axis-1]) > 1`. When Concat::evaluate() calls reference::concat(), at step k=1 the code reads `arg + k*(N/steps)` bytes into a buffer that is only `N/2` bytes. For example: two i4 inputs each shaped [2, 4] concatenated on axis=0 give steps=1 — but two inputs each shaped [1, 4] (N=4 elements, 2 bytes each) concatenated on axis=1 give steps=2, N/steps=2 elements/step, in_offset at step 1 = 2, but each source buffer is only 2 bytes (indices 0-1); the code reads from byte 2, past the end of the allocation.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the OOB read at
//   openvino/src/core/reference/src/op/concat.cpp:58-61
// Pre-fix: op::v0::Concat::evaluate on i4 inputs with steps>1 makes
// copy_elements read arg+step*(N/steps) bytes from an N/2-byte packed
// buffer (heap OOB read; flagged by ASan). Post-fix: in_offset is a true
// byte offset and the read stays in bounds, so evaluate succeeds and the
// concatenated result is correct.
//
// TODO: confirm exact i4 tensor/Constant construction API and the test
//       target/include paths against the existing core reference/op test
//       tree (e.g. src/core/tests/eval.cpp) before building.
#include <gtest/gtest.h>
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

TEST(eval_concat, i4_axis_gt0_no_oob_read) {
    // Two i4 inputs shaped [2,4] (N=8 elems = 4 packed bytes each),
    // concatenated on axis=1 -> out [2,8], steps = out_shape[0] = 2.
    const Shape in_shape{2, 4};
    // TODO: fill with valid packed i4 data; data() byte size must be 4.
    auto a = op::v0::Constant::create(element::i4, in_shape, std::vector<int>(8, 1));
    auto b = op::v0::Constant::create(element::i4, in_shape, std::vector<int>(8, 2));
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{a, b}, /*axis=*/1);

    Tensor out(element::i4, Shape{2, 8});
    TensorVector outputs{out};
    TensorVector inputs{a->get_tensor_view(), b->get_tensor_view()}; // TODO: adjust accessor
    // Pre-fix this evaluate() reads past the 4-byte i4 source buffers.
    ASSERT_TRUE(concat->evaluate(outputs, inputs));
}
```
**Build / run:** Build target ov_core_unit_tests (or the core reference/op eval test target). Run: ov_core_unit_tests --gtest_filter='eval_concat.i4_axis_gt0_no_oob_read'. Pre-fix, an ASan build reports 'heap-buffer-overflow READ' inside std::memcpy from ov::reference::copy_elements (concat.cpp:27) sourced at concat.cpp:58/61; post-fix the test passes with no ASan error.

## Suggested fix
Compute `in_offset` after halving `size`, or halve `in_offset` separately for nibble types: replace lines 57-61 with:
  size_t size = shape_sizes[in_index] / steps;
  if (elem_type == ov::element::u4 || elem_type == ov::element::i4) size /= 2;
  const size_t in_offset = step * size;  // now in bytes
  copy_func(args[in_index], out, in_offset, out_offset, size, elem_size);
This ensures in_offset is always a byte offset consistent with the packed buffer layout.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #378.
