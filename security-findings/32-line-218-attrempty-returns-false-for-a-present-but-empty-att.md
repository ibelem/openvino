# Security finding #32: Line 218: `attr.empty()` returns false for a present-but-empty attr…

**Summary:** Line 218: `attr.empty()` returns false for a present-but-empty attr…

**CWE IDs:** CWE-20: Improper Input Validation
**Severity / Impact:** Uncaught `std::invalid_argument` propagates out of `get_uint64_attr`, through every caller that passes a default value, causing a crash/DoS for any application that loads a crafted model. All three callers (`frontend.cpp:38`, `input_model.cpp:271`, and any others using the default-valued overload) are affected.
**Affected location:** `targets/openvino/src/common/util/src/xml_parse_utils.cpp:216` — `ov::util::pugixml::get_uint64_attr(const pugi::xml_node&, const char*, uint64_t)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted XML model file → pugixml attribute parsing → get_uint64_attr default-value overload

## Description / Root cause
Line 218: `attr.empty()` returns false for a present-but-empty attribute (e.g., `version=""`). pugixml's `xml_attribute::empty()` is a null/missing-attribute check, not a value-empty check. When the attribute exists with an empty string value, the guard at line 218 does NOT return the default, and control falls through to the mandatory single-arg overload at line 220. That overload calls `std::stoll("", &idx, 10)` at line 76, which throws `std::invalid_argument` because the empty string has no valid digits. The same defect exists in `get_int_attr` (line 202-207) and `get_int64_attr` (line 209-214) default-overloads for the same reason.

**Validator analysis:** The flaw is genuine: pugixml's xml_attribute::empty() (lines 218/211/204) only detects a missing attribute, not a present attribute with an empty string value. With `version=""`, the default-overload guard does not fire, control reaches the single-arg overload at line 220, and line 76 `std::stoll("", &idx, 10)` throws std::invalid_argument. The path frontend.cpp:38 get_ir_version → supported_impl (99-138) / load_impl (152-248) and input_model.cpp:271 has no try/catch, so the exception escapes the FrontEnd API. The vulnType (CWE-20 Improper Input Validation) is accurate. The impact (uncaught exception → crash/DoS on a crafted model) is plausible but somewhat overstated: ov::Core::read_model and the frontend manager typically wrap frontend probing in try/catch and re-throw as ov::Exception, so end-to-end this is more often a thrown-exception (rejected model) than a hard process abort — still a robustness/availability defect. The proposed fix is correct and sufficient: after the `attr.empty()` check, also return defVal when `std::string(attr.value()).empty()` for all three default-value overloads (get_int_attr 202-206, get_int64_attr 209-213, get_uint64_attr 216-220, and the analogous get_uint_attr 223-227). The defence-in-depth try/catch at callers is optional but reasonable. A cleaner alternative is to centralize the empty-value handling inside the single-arg overloads themselves (treat empty value as 0 or a clear runtime_error) so all callers are protected uniformly.

## Exploit / Proof of Concept
Craft a minimal XML file: `<net version="">...</net>`. Supply it to `FrontEnd::supported_impl()` or `FrontEnd::load_impl()`. `get_ir_version(doc)` → `get_uint64_attr(root, "version", 0)` → `attr.empty()` returns `false` (attribute IS present) → falls to mandatory overload → `std::stoll("")` → `std::invalid_argument` thrown. No try/catch exists anywhere on this path up through `supported_impl` (lines 99-138) or `load_impl` (lines 152-248), so the exception escapes the FrontEnd API, terminating or crashing the calling process.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
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
```
**Build / run:** Build target: ov_ir_frontend_tests (cmake --build . --target ov_ir_frontend_tests). Run: ./ov_ir_frontend_tests --gtest_filter=IRFrontendTests.empty_version_attribute_does_not_throw_invalid_argument . Expected pre-fix failure: an uncaught std::invalid_argument ("stoll") escapes getWithIRFrontend(), tripping the FAIL()/test-level exception; post-fix the empty value maps to default version 0 and the test passes (model is nullptr or a typed ov::Exception is caught).

## Suggested fix
In the default-value overload (lines 216-221), replace the `attr.empty()` guard with a check that also catches present-but-empty values: `if (attr.empty()) return defVal; auto str_value = std::string(attr.value()); if (str_value.empty()) return defVal;`. This prevents the fall-through to `std::stoll` on an empty string. Apply the same fix to the `get_int_attr` and `get_int64_attr` default-value overloads (lines 202-214). Additionally, wrap `get_uint64_attr` calls in `frontend.cpp:38` and `input_model.cpp:271` with a try/catch for `std::exception` to add defence-in-depth at the trust boundary.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #32.
