// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for CWE-369 divide-by-zero in
//   targets/openvino/src/plugins/intel_cpu/src/nodes/gather.cpp:205-206 and 217-218
// where scale_group_size / zp_group_size are computed by dividing
//   getInputShapeAtPort(GATHER_DATA).getElementsCount()
// by getInputShapeAtPort(GATHER_SCALE/ZP).getElementsCount(), which is 0
// for a static shape containing a zero-valued dimension (cpu_shape.h:165-173).
//
// Pre-fix: constructing a GatherCompressed node with a static {0} decompression
//   scale and compiling it on the CPU plugin triggers SIGFPE inside
//   Gather::initSupportedPrimitiveDescriptors (no clean throw).
// Post-fix: the node is rejected with an ov::Exception (CPU_NODE_ASSERT) instead.
//
// HARNESS: ov_cpu_unit_tests (intel_cpu component gtest target).
// TODO: confirm the exact target name and an existing test fixture by reading
//       targets/openvino/src/plugins/intel_cpu/tests/unit/ (could not enumerate
//       the directory with the read-only tool).

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/gather_compressed.hpp"

using namespace ov;

// TODO: verify GatherCompressed constructor signature against
//       targets/openvino/src/common/transformations/include/ov_ops/gather_compressed.hpp
TEST(GatherCompressedCpuRegression, ZeroElementScaleIsRejectedNotSigfpe) {
    // DATA {4,8} u8, INDICES {2} i32, AXIS const 0, SCALE {0} f32 (zero elements).
    auto data    = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{4, 8});
    auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    auto axis    = op::v0::Constant::create(element::i64, Shape{}, {0});
    // Zero-element static scale tensor -> getElementsCount() == 0.
    auto scale   = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});

    auto gather_compressed = std::make_shared<op::internal::GatherCompressed>(
        data, indices, axis, /*batch_dims=*/0, scale);

    auto model = std::make_shared<Model>(OutputVector{gather_compressed},
                                         ParameterVector{data, indices},
                                         "gather_compressed_zero_scale");

    ov::Core core;
    // Pre-fix this compilation path divides by zero in
    // Gather::initSupportedPrimitiveDescriptors -> SIGFPE.
    // Post-fix it must throw a catchable ov::Exception.
    EXPECT_THROW(core.compile_model(model, "CPU"), ov::Exception);
}
