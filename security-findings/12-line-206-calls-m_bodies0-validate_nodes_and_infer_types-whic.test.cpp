// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-674 in ov::op::v5::Loop::validate_and_infer_types
//   targets/openvino/src/core/src/op/loop.cpp:206 and :267
//   (m_bodies[0]->validate_nodes_and_infer_types() re-enters Loop validation
//    for every nested Loop with no recursion depth cap -> stack overflow).
//
// What this encodes:
//   - Build N deeply-nested Loop ops (each body contains a Loop).
//   - PRE-FIX: validation recurses N deep and overflows the stack (crash;
//     ASan/native SIGSEGV, NOT catchable by EXPECT_THROW).
//   - POST-FIX: the depth guard throws ov::Exception before exhausting the
//     stack, so the construction is rejected cleanly.
//
// Harness: ov_core_tests (gtest). Place in the nearest core op test dir,
// e.g. targets/openvino/src/core/tests/type_prop/loop.cpp
// TODO: confirm exact test file + target name by reading the surrounding
//       core tests tree before adding.

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

// TODO: factor a helper that returns a minimal Loop whose body is `inner`
//       (a trivial body when inner == nullptr). Exact SpecialBodyPorts /
//       trip-count / exec-condition wiring must be copied from an existing
//       passing Loop type_prop test — do not guess the port-map indices.
static std::shared_ptr<op::v5::Loop> make_nested_loop(size_t depth) {
    // TODO: implement nesting:
    //   for i in [0, depth): wrap the current body Model in a new Loop whose
    //   single body node is that Loop, with one trivial merged/back-edge input.
    //   Each level must add a back-edge so loop.cpp:267 is also exercised.
    (void)depth;
    return nullptr; // TODO
}

TEST(type_prop, loop_deeply_nested_recursion_is_bounded) {
    constexpr size_t kDepth = 4000;  // exceeds default thread stack pre-fix
    auto outer = make_nested_loop(kDepth);
    // POST-FIX expectation: a hard recursion-depth cap rejects the model
    // instead of overflowing the stack.
    EXPECT_THROW(outer->validate_and_infer_types(), ov::Exception);
}
