# Security finding #513: `sizeOD`, `sizeOH`, `sizeOW` are declared as `int` (lines 3445-3447…

**Summary:** `sizeOD`, `sizeOH`, `sizeOW` are declared as `int` (lines 3445-3447…

**CWE IDs:** CWE-190: Integer Overflow or Wraparound
**Severity / Impact:** Heap buffer overflow or heap exhaustion DoS. If the signed overflow wraps to a small positive value, `auxTable` is allocated too small, and the three write loops beginning at lines 3462, 3475, 3485 perform OOB writes. This is exploitable to corrupt heap metadata or adjacent allocations and may lead to code execution. Triggered during inference graph preparation with a malicious model.
**Affected location:** `targets/openvino/src/plugins/intel_cpu/src/nodes/interpolate.cpp:3445` — `Interpolate::InterpolateExecutorBase::buildTblLinear()`
**Validated for repos:** openvinoEp, openvino
**Trust boundary:** Model constant SCALES_ID tensor; attacker controls scale magnitude, which sets kernel radius rx/ry/rz

## Description / Root cause
`sizeOD`, `sizeOH`, `sizeOW` are declared as `int` (lines 3445-3447). The products `OD * diaOD`, `OH * diaOH`, `OW * diaOW` mix `size_t` (OD/OH/OW come from `dstDim5d`, a `VectorDims` i.e. `std::vector<size_t>`) with `int` (diaO*), and are then truncated into `int` variables. Even without triggering the +Inf UB, a legitimate but very small scale (e.g., 0.001) yields a large `rx` (e.g., `ceil(2/0.001)=2000`), `diaOW=4001`, and with a large output dimension OW=1048576, `sizeOW = 1048576 * 4001` overflows `int` (max ~2.1e9). The expression `(sizeOD + sizeOH + sizeOW) * 2` then passes a negative or wrapped value to `auxTable.resize()`, which takes `size_t`, reinterpreting the signed value as a huge unsigned number (or zero), causing either a catastrophic allocation or a near-zero buffer followed by OOB writes in the subsequent loop.

**Validator analysis:** CWE-190 is accurate: lines 3445-3447 compute `OD*diaOD` etc. where OD/OH/OW are size_t (from dstDim5d, a VectorDims) and diaO* are int; the product is evaluated in size_t but assigned to a 32-bit `int`, truncating any value > INT_MAX. The sum `(sizeOD+sizeOH+sizeOW)*2` is then int arithmetic passed to resize(size_t): if it wraps to a small/zero positive value the auxTable is undersized, and the subsequent write loops (bounded by the true size_t OD/OH/OW and diaO*) write far past the buffer → heap OOB; if it wraps negative the size_t conversion requests a gigantic allocation → DoS. There is no bounds/clamp guard before resize, and resize's size_t parameter is the mitigation that is NOT present. The impact (heap OOB write / heap-exhaustion DoS) is accurate; 'code execution' is speculative. One caveat to the exploit narrative: because radius rx≈kernel_width/fx and OW≈fx*IW, sizeOW≈4*IW, so triggering >INT_MAX practically needs an extreme single-axis dimension (hundreds of millions) — large but model-specifiable/allocatable, so still reachable. The proposed fix (promote dia*/size* to int64_t/size_t and validate the total table size against a sane maximum before resize) is correct and sufficient; computing the size in size_t and OPENVINO_THROW on an unreasonable magnitude closes both the truncation and the exhaustion paths.

## Exploit / Proof of Concept
Supply a model with `antialias=true`, scale=0.001, and output spatial dimensions large enough that `OW * (2 * ceil(kernel_width/0.001) + 1)` exceeds INT_MAX. E.g., kernel_width=2, scale=0.001 → rx=2000, diaOW=4001; OW=537396 → sizeOW=2.15e9 > INT_MAX → signed overflow. After resize, the loops at lines 3488-3495 write `OW * diaOW` entries past the end of `auxTable`.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: ov_cpu_unit_tests (intel_cpu unit tests). Run: ov_cpu_unit_tests --gtest_filter=InterpolateBuildTblLinearOverflow.* with an ASan-instrumented build. Pre-fix expected: AddressSanitizer 'heap-buffer-overflow WRITE' inside Interpolate::InterpolateExecutorBase::buildTblLinear (write loops near interpolate.cpp:3463/3489) due to the truncated auxTable.resize at line 3448. Post-fix expected: ov::Exception thrown (table size validated) and the test passes.

## Suggested fix
Change the types of `diaOD`, `diaOH`, `diaOW`, `sizeOD`, `sizeOH`, `sizeOW` from `int` to `int64_t` or `size_t`. Before calling `auxTable.resize()`, verify `(sizeOD + sizeOH + sizeOW)` does not exceed a reasonable maximum (e.g., 512 MB) and is positive. Example: `int64_t sizeOW = static_cast<int64_t>(OW) * diaOW; if (sizeOW <= 0 || sizeOW > MAX_TABLE_SIZE) OPENVINO_THROW(...);`


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #513.
