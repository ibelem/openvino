# Security finding #20: At line 413, `srcRowIdx = pIdx[r]` reads a signed int64_t from the …

**Summary:** At line 413, `srcRowIdx = pIdx[r]` reads a signed int64_t from the …

**CWE IDs:** CWE-129: Improper Validation of Array Index
**Severity / Impact:** Out-of-bounds read: std::copy_n at line 415 copies src_shape[1]*src_type.size() bytes from an arbitrary out-of-bounds host-memory address into pDst. This causes either a process crash (SIGSEGV/access violation) or — if the OOB address is mapped — an information leak of arbitrary host-process memory into the model's output tensor, readable by the model consumer.
**Affected location:** `targets/openvino/src/plugins/intel_npu/src/plugin/npuw/util.cpp:413` — `ov::npuw::util::gather()`
**Validated for repos:** openvino
**Trust boundary:** idx tensor values from model inference pipeline — untrusted row-index data passed in via handle_quant_host_gather() in base_sync_infer_request.cpp:562,572,573,585

## Description / Root cause
At line 413, `srcRowIdx = pIdx[r]` reads a signed int64_t from the caller-supplied `idx` tensor. At line 414, this value is immediately used in pointer arithmetic `pSrc + src_shape[1] * srcRowIdx * src_type.size()` with no preceding check that `0 <= srcRowIdx < (int64_t)src_shape[0]`. The signed `int64_t` is implicitly converted to `size_t` (unsigned 64-bit) in the multiply expression; a negative srcRowIdx wraps to a huge unsigned value causing the pointer to land far before `pSrc`, and any value >= src_shape[0] causes it to land past the allocation. No validation exists anywhere in the function or at any call site in base_sync_infer_request.cpp.

**Validator analysis:** Confirmed. In ov::npuw::util::gather (util.cpp:386-418) the NPUW_ASSERTs at 391-406 only validate element types and shape rank/dims, never the index VALUES. At line 413 srcRowIdx=pIdx[r] is read from the idx tensor (i64) and at 414 used as pSrc + src_shape[1]*srcRowIdx*src_type.size(); the signed value is implicitly widened to size_t, so a negative index wraps to a huge offset and any index>=src_shape[0] lands past the allocation, then std::copy_n at 415 performs an OOB read of src_shape[1]*elem_size bytes into the output tensor. CWE-129 is accurate; impact (OOB read → crash or host-memory info-leak into the model output) is accurate. The path is reachable: handle_quant_host_gather() passes the submodel input-port tensor 'lookup' (request->get_tensor(lport)) straight to gather() at 562/572/573/585 with no value-range sanitization. The proposed fix is correct and sufficient: check srcRowIdx<0 || (size_t)srcRowIdx>=src_shape[0] and OPENVINO_THROW before casting to size_t. Minor refinement: gather_cb4 (435-449) has the same shape of issue with u4 indices bounded to src size, but it is constrained to 0..15 by the 4-bit width and a 16-entry src is the documented layout, so it is lower risk; the fix here should at least cover the i64 gather. openvinoEp: the defect cannot propagate there — the EP layer has no gather call — but since the EP can in principle feed model inputs into an NPU compiled model, this is a boundary-reachability rejection rather than na; marked rejected because no EP code in plugin_impl exercises this path.

## Exploit / Proof of Concept
An attacker who controls the model's quantized gather index input (e.g. input_ids / vocab lookup indices fed to the prefill or generate inference request) can supply a negative int64_t index such as -1 or INT64_MIN. In handle_quant_host_gather() (base_sync_infer_request.cpp:562) the lookup tensor is retrieved directly from the compiled submodel's input port and passed to util::gather() without any value-range sanitization. Inside gather(), srcRowIdx=-1 causes the multiplication src_shape[1] * (-1) * elem_size to produce a huge uint64_t offset, making pSrcRow point to memory far below the src allocation. std::copy_n then reads from that location, crashing or leaking memory contents.

## Reproduction
_(not provided)_

## Test (gtest-ov)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-129 OOB read in ov::npuw::util::gather
//   src: openvino/src/plugins/intel_npu/src/plugin/npuw/util.cpp:413-415
//        auto srcRowIdx = pIdx[r];
//        auto pSrcRow = pSrc + src_shape[1] * srcRowIdx * src_type.size();   // <-- no bounds check
//        std::copy_n(pSrcRow, src_shape[1] * src_type.size(), pDst);
//
// Pre-fix: a negative or >=src_shape[0] index makes pSrcRow point outside the
//   src allocation; std::copy_n reads OOB -> ASan heap-buffer-overflow (or SIGSEGV).
// Post-fix (proposed): gather() validates 0 <= idx < src_shape[0] and throws
//   ov::Exception, so EXPECT_THROW(..., ov::Exception) passes and no OOB occurs.
//
// Harness: ov_npu_unit_tests (gtest + ASan). util.cpp is already in OBJECT_FILES
// of tests/unit/CMakeLists.txt, and util.hpp is on the include path.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace {

// src: 4 rows x 8 cols of f16; valid index range is [0, 4).
static ov::SoPtr<ov::ITensor> make_src() {
    ov::Tensor t(ov::element::f16, ov::Shape{4, 8});
    std::fill_n(static_cast<uint8_t*>(t.data()), t.get_byte_size(), uint8_t{0});
    return ov::get_tensor_impl(t);
}

// idx: shape {1, N} of i64 carrying a single attacker-controlled lookup value.
static ov::SoPtr<ov::ITensor> make_idx(int64_t value) {
    ov::Tensor t(ov::element::i64, ov::Shape{1, 1});
    t.data<int64_t>()[0] = value;
    return ov::get_tensor_impl(t);
}

// dst: shape {1, N, 8} of f16 (src_shape[1] == dst_shape[2]).
static ov::SoPtr<ov::ITensor> make_dst() {
    ov::Tensor t(ov::element::f16, ov::Shape{1, 1, 8});
    return ov::get_tensor_impl(t);
}

// A valid, in-range index must succeed.
TEST(NpuwUtilGatherBounds, InRangeIndexSucceeds) {
    EXPECT_NO_THROW(ov::npuw::util::gather(make_src(), make_idx(2), make_dst()));
}

// Negative index: pre-fix wraps to a huge size_t offset -> OOB read (ASan).
// Post-fix: rejected with ov::Exception.
TEST(NpuwUtilGatherBounds, NegativeIndexRejected) {
    EXPECT_THROW(ov::npuw::util::gather(make_src(), make_idx(-1), make_dst()), ov::Exception);
}

// Index >= src_shape[0]: pre-fix reads past the src allocation (ASan).
// Post-fix: rejected with ov::Exception.
TEST(NpuwUtilGatherBounds, OverflowIndexRejected) {
    EXPECT_THROW(ov::npuw::util::gather(make_src(), make_idx(4), make_dst()), ov::Exception);
}

}  // namespace
```
**Build / run:** Add the file as targets/openvino/src/plugins/intel_npu/tests/unit/npuw/gather_bounds_test.cpp and register it in tests/unit/CMakeLists.txt target_sources(ov_npu_unit_tests ...). Build target: ov_npu_unit_tests (configure with -DENABLE_SANITIZER=ON for ASan). Run: ov_npu_unit_tests --gtest_filter='NpuwUtilGatherBounds.*'. Pre-fix expectation: NegativeIndexRejected / OverflowIndexRejected trigger an AddressSanitizer 'heap-buffer-overflow READ' inside ov::npuw::util::gather (std::copy_n at util.cpp:415) — or a SIGSEGV — instead of throwing. Post-fix: all three tests pass (in-range succeeds, out-of-range throw ov::Exception).

## Suggested fix
Add an explicit bounds check inside the loop before the pointer computation:

  for (std::size_t r = 0; r < idx_shape[1]; r++) {
      auto srcRowIdx = pIdx[r];
      if (srcRowIdx < 0 || static_cast<std::size_t>(srcRowIdx) >= src_shape[0]) {
          OPENVINO_THROW("gather: index ", srcRowIdx, " out of range [0, ", src_shape[0], ")");
      }
      auto pSrcRow = pSrc + src_shape[1] * static_cast<std::size_t>(srcRowIdx) * src_type.size();
      ...
  }

The cast to size_t should only happen after the sign check to avoid the implicit signed-to-unsigned conversion.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #20.
