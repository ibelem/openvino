// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190/CWE-125 in
//   openvino/src/core/xml_util/src/xml_deserialize_util.cpp:893-895
//   (XmlDeserializer::set_constant_num_buffer)
// Pre-fix: offset=0xFFFFFFFFFFFFFFFF + size=1 wraps to 0, the bounds
//   OPENVINO_ASSERT at line 895 passes, then get_ptr<char>()+offset (line 897)
//   is dereferenced out-of-bounds -> ASan heap-buffer-overflow / read.
// Post-fix: the input is rejected and read_model throws ov::Exception.
//
// Harness: IR frontend tests (ov_core_unit_tests / IR frontend test target).
// TODO: confirm the exact target & include path by reading the nearest
//       existing IR-frontend test dir (e.g. src/frontends/ir/tests/) and
//       mirror its read_model(model_str, weights_tensor) helper.
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

TEST(IRFrontendOverflow, ConstOffsetSizeWraparoundRejected) {
    // Const with offset = SIZE_MAX, size = 1; offset+size wraps to 0 pre-fix.
    std::string model = R"V0G0N(<?xml version="1.0"?>
<net name="oob" version="11">
    <layers>
        <layer id="0" name="c" type="Const" version="opset1">
            <data element_type="i8" shape="1" offset="18446744073709551615" size="1"/>
            <output>
                <port id="0" precision="I8"><dim>1</dim></port>
            </output>
        </layer>
        <layer id="1" name="r" type="Result" version="opset1">
            <input><port id="0" precision="I8"><dim>1</dim></port></input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>)V0G0N";

    // Tiny, valid weights buffer so m_weights is non-null and small.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});
    std::memset(weights.data(), 0, weights.get_byte_size());

    ov::Core core;
    // TODO: if Core::read_model(std::string, ov::Tensor) overload differs in
    //       this tree, switch to the IR test fixture's createModelFromString().
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
