// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-190 in openvino/src/frontends/ir/src/input_model.cpp:136
// (parse_pre_process). Pre-fix: a crafted <mean offset=HUGE size=N> where
// offset+size wraps past UINT64_MAX bypasses the `const_offset + const_size >
// weights->size()` check, so line 186 forms an out-of-bounds pointer and
// Constant::create memcpy's from it (ASan heap-buffer-overflow / SIGSEGV).
// Post-fix (offset checked against weights->size()-const_size): convert_model
// must throw ov::Exception instead.
//
// Goes in the IR frontend gtest target (ov_ir_frontend_tests), e.g.
// src/frontends/ir/tests/frontend_test_basic.cpp style.
#include <gtest/gtest.h>
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/core/except.hpp"

using namespace ov;

TEST(IRFrontendPreProcess, MeanOffsetOverflowRejected) {
    // TODO: build a minimal IR <net> with a 4D NCHW Parameter (shape [1,3,H,W])
    //       and a <pre-process><channel id=..><mean size='N' offset='OFF'/> block
    //       for every channel, where:
    //         N   = shape_size({1,H,W}) * sizeof(float)   (passes line 128)
    //         OFF = 18446744073709551000ULL                (so OFF + N wraps < N)
    //       Pick H=W small so N is tiny and the wrapped sum < weights size.
    std::string xml = R"(<!-- TODO: crafted IR XML as described above --> )";

    // TODO: allocate a weights AlignedBuffer larger than the wrapped sum so the
    //       bypassed check would otherwise pass; e.g. shared_ptr<AlignedBuffer>.
    auto weights = std::make_shared<ov::AlignedBuffer>(/*size=*/4096);

    ov::frontend::FrontEndManager fem;
    auto fe = fem.load_by_framework("ir");
    ASSERT_NE(fe, nullptr);
    // TODO: load via the IR FE's stream+weights input variant used by the
    //       existing ov_ir_frontend_tests fixtures.
    auto in_model = fe->load(/* TODO: stream(xml) + weights */);
    // Expect rejection of the overflowing offset rather than an OOB read.
    EXPECT_THROW(fe->convert(in_model), ov::Exception);
}
