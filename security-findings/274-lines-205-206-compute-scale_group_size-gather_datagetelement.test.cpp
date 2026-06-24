// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-369 in gather.cpp:205-206 (scale_group_size) and 217-218 (zp_group_size).
// Pre-fix: a GatherCompressed node whose SCALE tensor has MORE elements than the DATA tensor makes
//   scale_group_size = dataElems / scaleElems truncate to 0, then divided at gather.cpp:832-833 -> SIGFPE/UB.
// Post-fix: initSupportedPrimitiveDescriptors() must reject the node (CPU_NODE_THROW) before execution.
//
// Harness: ov_cpu_unit_tests (OpenVINO intel_cpu component tests).
// TODO: confirm exact test dir/helpers by reading openvino/src/plugins/intel_cpu/tests/unit/ and the
//       existing GatherCompressed/node single-layer test that builds an ov::op::internal::GatherCompressed.

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
// TODO: include the internal GatherCompressed op header used by the CPU plugin
// #include "ov_ops/gather_compressed.hpp"

// TODO: include the CPU-node test fixture that lets you build a single node and call
//       initSupportedPrimitiveDescriptors()/prepareParams()/execute() (see intel_cpu/tests/unit).

TEST(GatherCompressedCpuNode, RejectsScaleWithMoreElementsThanData_NoDivByZero) {
    // data: shape [4] -> 4 elements ; scale: shape [8] -> 8 elements (hostile, scale_group_size would be 0)
    // TODO: build Parameter(data u8 [4]), indices i32 [1], axis const 0, scale f32 [8] (+ optional zp f32 [8]),
    //       wrap into ov::op::internal::GatherCompressed, construct the CPU Gather node with a GraphContext,
    //       then assert the node refuses the configuration once the fix is applied.
    //
    // EXPECT_THROW(node.initSupportedPrimitiveDescriptors(), ov::Exception);   // passes only after fix
    //
    // Without the fix, execution of execCompressed8Bit() at gather.cpp:832 divides by scale_group_size==0,
    // which ov_cpu_unit_tests under ASan/UBSan reports as SIGFPE / 'division by zero'.
    GTEST_SKIP() << "TODO: needs GatherCompressed node builder + CPU node fixture wiring (see intel_cpu/tests/unit).";
}
