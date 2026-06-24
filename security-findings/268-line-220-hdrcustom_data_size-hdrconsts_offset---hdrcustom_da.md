# Security finding #268: Line 220: `hdr.custom_data_size == hdr.consts_offset - hdr.custom_d…

**Summary:** Line 220: `hdr.custom_data_size == hdr.consts_offset - hdr.custom_d…

**CWE IDs:** CWE-191: Integer Underflow (Wrap or Wraparound)
**Severity / Impact:** Remote/local attacker supplying a crafted model cache blob triggers an enormous std::string::resize() call (line 232), causing std::bad_alloc or process OOM crash. This is a reliable Denial-of-Service against any OpenVINO CPU plugin load of a malicious cached model, using only a ~50-byte blob.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:220` — `ModelDeserializer::process_model (istream overload)()`
**Validated for repos:** openvino
**Trust boundary:** Untrusted DataHeader read raw from cache stream at line 216; all size_t fields attacker-controlled

## Description / Root cause
Line 220: `hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset` performs unsigned subtraction without first verifying `hdr.consts_offset >= hdr.custom_data_offset`. If an attacker sets `hdr.consts_offset = 0` (< sizeof(hdr)=48), the subtraction wraps to 0xFFFFFFFFFFFFFFD0. An attacker can also set `hdr.custom_data_size` to that same wrapped value, so the equality check passes. The validation flag `is_valid_model` becomes true, OPENVINO_ASSERT at line 222 does not fire, and then `xmlInOutString.resize(hdr.custom_data_size)` at line 232 is called with the ~2^64 byte value.

**Validator analysis:** The flaw is real and matches the cited code exactly. DataHeader is six size_t fields (sizeof=48), all read raw from the stream at line 216, so every field is attacker-controlled. Line 219-221 validates relationships via unsigned subtraction without first requiring consts_offset >= custom_data_offset or model_offset >= consts_offset. Because both operands of each equality are attacker-controlled, the attacker matches the wrapped subtraction result, so is_valid_model is true and the OPENVINO_ASSERT at 222 does not fire. Line 232 then calls std::string::resize(0xFFFFFFFFFFFFFFD0). CWE-191 (integer underflow) is the correct root cause. The impact label 'OOM/bad_alloc' is slightly imprecise: a value > string::max_size throws std::length_error (not bad_alloc); a moderately large but allocatable wrapped value would instead cause a multi-GB alloc / bad_alloc. Either way it is an uncaught-exception/DoS if not wrapped by a try/catch (none seen on this path), so the DoS claim stands. The proposed fix (add consts_offset>=custom_data_offset and model_offset>=consts_offset monotonicity guards into is_valid_model) is correct and sufficient to make every subtraction non-wrapping before it is used; recommend also keeping file_size>=hdr.model_offset (current uses >) consistent. For the EP repo the defect is not in EP-owned code and is not reachable from the ONNX model trust boundary, only from a crafted cache file on disk, so it is rejected there.

## Exploit / Proof of Concept
Craft a DataHeader blob: custom_data_offset=48 (sizeof DataHeader), consts_offset=0, custom_data_size=0xFFFFFFFFFFFFFFD0 (= 0-48 unsigned), model_offset=49, consts_size=49, plus file_size>49. Conditions (a) 48==48, (b) 0xFFFFFFFFFFFFFFD0==0-48, (c) 49==49-0, (d) file_size>49 all pass. OPENVINO_ASSERT passes. At line 232: `xmlInOutString.resize(0xFFFFFFFFFFFFFFD0)` exhausts process memory and terminates the host application.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-191 integer underflow in
//   openvino/src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp:219-232
// Pre-fix: a crafted DataHeader with consts_offset=0, custom_data_offset=48,
//   custom_data_size=0-48 (=0xFFFFFFFFFFFFFFD0) passes the equality check at line 220,
//   OPENVINO_ASSERT(222) does not fire, and resize(line 232) throws std::length_error
//   (or attempts a huge allocation) -> DoS / wrong exception type.
// Post-fix: monotonicity guard makes is_valid_model false, so OPENVINO_ASSERT throws ov::Exception.
//
// NOTE: ModelDeserializer / pass::StreamSerialize::DataHeader are internal CPU-plugin
// types; the exact include path and ctor signature must be confirmed against the source.
#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
#include "openvino/core/except.hpp"
// TODO: include the real headers exposing ModelDeserializer and StreamSerialize::DataHeader
//       (e.g. src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.hpp and
//        the serialize pass header). Confirm names before compiling.

TEST(CpuModelDeserializer, MalformedHeaderRejectedNoUnderflow) {
    // 6 x size_t header laid out as in StreamSerialize::DataHeader
    struct DataHeaderLayout {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    } hdr{};
    static_assert(sizeof(DataHeaderLayout) == 48, "adjust to real DataHeader layout");

    hdr.custom_data_offset = sizeof(DataHeaderLayout);          // 48
    hdr.consts_offset      = 0;                                 // underflow trigger
    hdr.custom_data_size   = (size_t)0 - sizeof(DataHeaderLayout); // 0xFFFFFFFFFFFFFFD0
    hdr.model_offset       = 49;
    hdr.consts_size        = 49;                                // == model_offset - consts_offset
    hdr.model_size         = 0;

    std::string blob(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    blob.resize(64, '\0'); // file_size > model_offset
    std::stringstream ss(blob, std::ios::in | std::ios::out | std::ios::binary);

    std::shared_ptr<ov::Model> model;
    // TODO: construct ModelDeserializer with the real ctor (decrypt config / model_buffer args)
    //       and invoke the istream overload of process_model(model, std::ref(ss)).
    // Expected POST-FIX behaviour: the monotonicity guard rejects the header.
    // EXPECT_THROW({ ModelDeserializer d(/*...*/); d.process_model(model, std::ref(ss)); }, ov::Exception);
    GTEST_SKIP() << "Fill in ModelDeserializer construction/invocation per source; assert ov::Exception is thrown (not std::length_error/bad_alloc).";
}
```
**Build / run:** Build target: ov_cpu_unit_tests. Run: ov_cpu_unit_tests --gtest_filter='CpuModelDeserializer.MalformedHeaderRejectedNoUnderflow'. Pre-fix the crafted blob reaches deserializer.cpp:232 std::string::resize(0xFFFFFFFFFFFFFFD0) -> std::length_error / huge allocation (ASan: allocation-size-too-big or bad_alloc abort); post-fix it throws ov::Exception from OPENVINO_ASSERT at line 222. Resolve the // TODO include/ctor lines against the real ModelDeserializer API before compiling.

## Suggested fix
Before the `is_valid_model` computation, add monotonicity checks: require `hdr.consts_offset >= hdr.custom_data_offset` and `hdr.model_offset >= hdr.consts_offset` before doing the subtractions, and add them to `is_valid_model`. E.g.: `bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr)) && (hdr.consts_offset >= hdr.custom_data_offset) && (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) && (hdr.model_offset >= hdr.consts_offset) && (hdr.consts_size == hdr.model_offset - hdr.consts_offset) && (file_size > hdr.model_offset);`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #268.
