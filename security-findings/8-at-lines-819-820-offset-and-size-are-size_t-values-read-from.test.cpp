// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for integer-overflow OOB read in
//   openvino/src/core/xml_util/src/xml_deserialize_util.cpp:828-830
// Pre-fix: offset=SIZE_MAX, size=1 makes `offset + size` wrap to 0, the
//   `m_weights->size() < offset+size` guard is bypassed, and
//   get_ptr<char>()+SIZE_MAX is read by unpack_string_tensor -> ASan OOB.
// Post-fix (split check `offset > size() || size > size()-offset`):
//   the malformed Const is rejected with ov::Exception instead.
//
// SKELETON: building a minimal valid IR XML + matching .bin/weights tensor
// that reaches the StringAlignedBuffer 'value'/'Const' branch requires a
// crafted fixture; the exact opset/string-const schema must be copied from a
// known-good string-Const IR. TODOs below name what is missing.

#include <gtest/gtest.h>
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(IRFrontend, string_const_offset_overflow_is_rejected) {
    // TODO: replace with a minimal IR whose op type="Const" carries
    //   <data element_type="string" shape="1"
    //         offset="18446744073709551615" size="1"/>
    //   so on_adapter() takes the StringAlignedBuffer branch (line 812-833).
    //   Confirm exact attribute/opset names against an existing string-Const
    //   IR test fixture under src/frontends/ir/tests.
    const std::string model = R"V0G0N(
<net name="overflow" version="11">
    <layers>
        <!-- TODO: parameter + string Const(offset=SIZE_MAX,size=1) + result -->
    </layers>
    <edges/>
</net>
)V0G0N";

    // Non-empty weights so m_weights is set (line 826) but small enough that a
    // real SIZE_MAX offset is wildly out of bounds.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});

    ov::Core core;
    // Pre-fix: ASan heap-buffer-overflow inside unpack_string_tensor.
    // Post-fix: clean throw of "Incorrect weights in bin file!".
    EXPECT_THROW(core.read_model(model, weights), ov::Exception);
}
