// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for input_model.cpp:136 (anonymous parse_pre_process).
// Unchecked uint64_t addition `const_offset + const_size > weights->size()` wraps to 0
// when const_offset = UINT64_MAX - const_size + 1, bypassing the bounds guard and causing
// `weights->get_ptr<char>() + offset` (line 186) to point ~18 EB out of bounds.
// Pre-fix: ASan heap-buffer-overflow / segfault (or silent OOB) during read_model.
// Post-fix (overflow-safe check): read_model throws before any pointer arithmetic.
//
// TODO: confirm target name is `ov_ir_frontend_tests` and the helper for reading a model
//       from an in-memory string + weights Tensor matches this repo's IR frontend test tree
//       (e.g. src/frontends/ir/tests/). Adjust input shape/type so that
//       shape_size(mean_shape)*element_size == CONST_SIZE to satisfy the line-128 check.
#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"

TEST(IRFrontendPreProcess, MeanOffsetIntegerOverflowIsRejected) {
    // input: f32, shape [1,1,1,1] => element_size=4, shape_size(mean_shape)=1 => const_size must be 4
    constexpr uint64_t CONST_SIZE = 4;
    // offset chosen so offset + size wraps to 0 in uint64_t
    const uint64_t bad_offset = UINT64_MAX - CONST_SIZE + 1ULL; // == 0xFFFFFFFFFFFFFFFC

    std::string model = R"V0G0N(<?xml version="1.0"?>
<net name="overflow" version="10">
  <layers>
    <layer id="0" name="in" type="Parameter" version="opset1">
      <data shape="1,1,1,1" element_type="f32"/>
      <pre-process reference-layer-name="in">
        <channel id="0"><mean size=")V0G0N" + std::to_string(CONST_SIZE) +
        R"V0G0N(" offset=")V0G0N" + std::to_string(bad_offset) +
        R"V0G0N("/></channel>
      </pre-process>
      <output><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>1</dim><dim>1</dim></port></output>
    </layer>
    <layer id="1" name="res" type="Result" version="opset1">
      <input><port id="0" precision="FP32"><dim>1</dim><dim>1</dim><dim>1</dim><dim>1</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
  </edges>
</net>)V0G0N";

    // Small weights buffer; any in-bounds offset would need to be < weights size.
    ov::Tensor weights(ov::element::u8, ov::Shape{CONST_SIZE});
    std::memset(weights.data(), 0, CONST_SIZE);

    ov::Core core;
    // Pre-fix the wrapped guard passes and OOB pointer math runs; post-fix this must throw.
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
