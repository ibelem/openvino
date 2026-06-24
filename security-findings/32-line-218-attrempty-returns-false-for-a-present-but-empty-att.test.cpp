// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-20 in
//   openvino/src/common/util/src/xml_parse_utils.cpp:216-221
//   ov::util::pugixml::get_uint64_attr(node, "version", defVal)
// Reached via:
//   openvino/src/frontends/ir/src/frontend.cpp:38  get_ir_version() -> get_uint64_attr(root,"version",0)
// Flaw: attr.empty() is false for a present-but-empty value (version=""), so control
// falls through to the mandatory overload -> std::stoll("") -> std::invalid_argument.
//
// Pre-fix: getWithIRFrontend() propagates an uncaught std::invalid_argument (test FAILS).
// Post-fix: empty value yields default version 0 -> model is simply unsupported and the
//           frontend returns nullptr without throwing std::invalid_argument (test PASSES).
//
// Place this TEST_F in src/frontends/ir/tests/frontend_test_basic.cpp (fixture IRFrontendTests,
// which already provides getWithIRFrontend()).

TEST_F(IRFrontendTests, empty_version_attribute_does_not_throw_invalid_argument) {
    std::string testModelEmptyVersion = R"V0G0N(
<net name="Network" version="">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges/>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;
    // The key assertion: parsing must not crash with an uncaught std::invalid_argument
    // from std::stoll(""). After the fix, the empty version resolves to default 0,
    // the IR is treated as unsupported, and getWithIRFrontend() returns nullptr cleanly.
    try {
        model = getWithIRFrontend(testModelEmptyVersion);
    } catch (const std::invalid_argument& e) {
        FAIL() << "empty version attribute leaked std::invalid_argument from std::stoll: " << e.what();
    } catch (const ov::Exception&) {
        // Acceptable: a controlled, typed rejection at the API boundary is fine.
        SUCCEED();
        return;
    }
    // If no throw, an unsupported/zero-version model must simply not be built.
    EXPECT_FALSE(!!model);
}