// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-190 integer overflow in
//   openvino/src/frontends/ir/src/input_model.cpp:136 (parse_pre_process)
// Pre-fix: `const_offset + const_size > weights->size()` wraps in uint64_t when
//   offset = UINT64_MAX - const_size + 1, bypassing the bounds check, leading to an
//   OOB pointer at lines 185-187 (caught by ASan as a heap-buffer-overflow read).
// Post-fix (subtraction-form guard): the crafted offset is rejected and the IR
//   frontend throws ov::Exception ("mean value offset and size are out of weights size range").
//
// Harness: ov_ir_frontend_tests (gtest + ASan), IR frontend test tree
//          openvino/src/frontends/ir/tests/.
//
// TODO(symbols): confirm the exact helper used by the existing IR frontend tests to
//   build a model from an in-memory xml string + weights buffer. The IR tests in
//   src/frontends/ir/tests typically use ov::Core::read_model(xml_string, weights_tensor)
//   or a FrontEndManager load_impl helper -- read the neighbouring test .cpp before use.
// TODO(shape): pick input/mean shapes so that shape_size(mean_shape)*type.size() == kConstSize
//   to satisfy the size-match guard at input_model.cpp:128 (here NCHW [1,3,2,2], f32 -> 16 bytes/chan).

#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"

TEST(IRFrontendPreProcessMean, MeanOffsetIntegerOverflowIsRejected) {
    // const_size for one channel of [1,1,2,2] f32 = 4 elems * 4 bytes = 16.
    constexpr uint64_t kConstSize = 16;
    // Wrapping offset: const_offset + const_size overflows uint64_t back to 0.
    constexpr uint64_t kOverflowOffset = 0xFFFFFFFFFFFFFFF0ull; // UINT64_MAX - 16 + 1

    const std::string xml = R"V0G0N(
<net name="overflow_mean" version="11">
  <pre-process reference-layer-name="in">
    <channel id="0"><mean offset=")V0G0N" + std::to_string(kOverflowOffset) +
        R"V0G0N(" size="16"/></channel>
  </pre-process>
  <layers>
    <layer id="0" name="in" type="Parameter" version="opset1">
      <data shape="1,1,2,2" element_type="f32"/>
      <output><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim></port></output>
    </layer>
    <layer id="1" name="res" type="Result" version="opset1">
      <input><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim></port></input>
    </layer>
  </layers>
  <edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/></edges>
</net>)V0G0N";

    // Small weights buffer -- a correct check must reject the wrapping offset, NOT
    // pass it through to weights->get_ptr<char>() + kOverflowOffset.
    ov::Tensor weights(ov::element::u8, ov::Shape{64});

    ov::Core core;
    // Pre-fix: ASan heap-buffer-overflow READ inside Constant::create (input_model.cpp:187).
    // Post-fix: deterministic ov::Exception from the bounds guard (input_model.cpp:136-137).
    EXPECT_THROW(core.read_model(xml, weights), ov::Exception);
}
