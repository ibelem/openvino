// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for interpolate.cpp:3445-3448 (Interpolate::InterpolateExecutorBase::buildTblLinear)
// Pre-fix: size_t products OD*diaOD / OH*diaOH / OW*diaOW are truncated into 32-bit `int`
// (lines 3445-3447). For dims/scales that push the product > INT_MAX, the int value
// wraps, auxTable.resize((sizeOD+sizeOH+sizeOW)*2) under-allocates (or requests a
// gigantic alloc), and the write loops at 3459/3472/3485 OOB-write. ASan should flag
// the heap-buffer-overflow pre-fix; post-fix the node must reject the model (throw).
//
// NOTE: this requires constructing an Interpolate (linear, antialias=true) op with a
// crafted scales constant and a huge output spatial dim. The exact intel_cpu unit-test
// harness symbols (graph/node fixture helpers) must be confirmed from the surrounding
// tests under openvino/src/plugins/intel_cpu/tests/unit/ before use.
//
// TODO: confirm the intel_cpu unit-test target name and the node/graph construction
//       helper used to instantiate an Interpolate node in isolation.
// TODO: confirm how to inject dataScales (SCALES_ID constant) and dstDim5d so that
//       OW * (2*ceil(kernel_width/scale)+1) exceeds INT_MAX.
#include <gtest/gtest.h>
#include "openvino/core/except.hpp"

TEST(InterpolateBuildTblLinearOverflow, RejectsTableSizeOverflow) {
    // TODO: build srcDimPad5d / dstDim5d with OW chosen so OW*diaOW > INT_MAX,
    //       dataScales with a tiny fx (antialias on) to inflate the kernel radius rx.
    // TODO: invoke the Interpolate executor build (linear+antialias) on these dims.
    // Post-fix expectation: the oversized auxTable size is detected and rejected.
    // EXPECT_THROW(build_interpolate_linear_executor(/*OW huge*/, /*fx tiny*/),
    //              ov::Exception);
    GTEST_SKIP() << "Fill in intel_cpu Interpolate node construction helpers; see TODOs.";
}
