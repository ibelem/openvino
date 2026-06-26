// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes regression for CWE-190 integer overflow at
// openvino/src/frontends/ir/src/input_model.cpp:136 (parse_pre_process).
// Pre-fix: const_offset + const_size wraps, bypassing the bounds check, and
// line 186 builds an OOB pointer that Constant::create reads -> ASan heap-buffer-overflow / SIGSEGV.
// Post-fix (split check): the IR frontend rejects the model with ov::Exception.
//
// TODO: this needs a crafted IR fixture; emitted as a skeleton.
#include <gtest/gtest.h>
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

TEST(IRFrontend, PreProcessMeanOffsetIntegerOverflowRejected) {
    ov::Core core;

    // TODO: build a minimal IR <net> whose input feeds a <pre-process> with a
    // <channel><mean size="4" offset="18446744073709551613"/></channel> so that
    // (offset + size) wraps below weights->size(). size must equal
    // shape_size(mean_shape)*input_type.size() to pass the line 128 check.
    const std::string ir_xml = R"V0G0N(
        <!-- TODO: full IR graph with <pre-process><channel id="0"><mean size="4" offset="18446744073709551613"/></channel></pre-process> -->
    )V0G0N";

    // TODO: weights buffer large enough that the *wrapped* sum (e.g. 1) <= size,
    // but the real offset is far out of bounds. Use a Tensor of a few bytes.
    ov::Tensor weights(ov::element::u8, ov::Shape{16});

    // Pre-fix: read_model proceeds and triggers an OOB read building the mean Constant.
    // Post-fix: the overflow-safe split check throws.
    EXPECT_THROW(core.read_model(ir_xml, weights), ov::Exception);
}
