// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.h"

#include <partitioned_mem_blk.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "edge.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/cpu_convert.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/cc/selective_build.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "selective_build.h"
#include "shape_inference/custom/gather.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>

#    include "kernels/x64/gather_uni_kernel.hpp"
#endif

using namespace dnnl::impl::cpu;

namespace ov::intel_cpu::node {

bool Gather::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gather_compression = ov::as_type_ptr<const ov::op::internal::GatherCompressed>(op);
        if (gather_compression) {
            return true;
        }

        if (op->get_output_element_type(0) == element::string) {
            return false;
        }
        if (none_of(op->get_type_info(),
                    ov::op::v7::Gather::get_type_info_static(),
                    ov::op::v8::Gather::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 7 and 8 versions.";
            return false;
        }

        if (!isDynamicNgraphNode(op) && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
            errorMessage = "Only Constant operation on 'axis' input is supported for static node.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Gather::Gather(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (any_of(op->get_input_size(), 4U, 5U) && op->get_output_size() == 1U) {
        compressed = true;
    } else {
        CPU_NODE_ASSERT(op->get_input_size() == 3 && op->get_output_size() == 1,
                        "has incorrect number of input/output edges!");
    }

    const auto& dataShape = getInputShapeAtPort(GATHER_DATA);
    isDataShapeStat = dataShape.isStatic();
    dataSrcRank = dataShape.getRank();

    const auto& idxShape = getInputShapeAtPort(GATHER_INDICES);
    isIdxShapeStat = idxShape.isStatic();
    const auto indicesRank = idxShape.getRank();
    CPU_NODE_ASSERT(dataSrcRank != 0LU && indicesRank != 0LU, "has incorrect input parameters ranks.");

    if (ov::is_type<ov::op::v8::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v8::Gather>(op)->get_batch_dims());
        // WA for NMS->Gather construction. NMS fills part of the output blob by the -1 if these values
        // must not be taken into account. There is appropriate pass that looks for such subgraphs
        // and sets the dontReverseIndices flag.
        const auto& rti = op->get_rt_info();
        const auto& reverse = rti.find("dontReverseIndices");
        reverseIndexing = reverse == rti.end();
    } else if (ov::is_type<ov::op::v7::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(op)->get_batch_dims());
        reverseIndexing = false;
    } else if (ov::is_type<ov::op::internal::GatherCompressed>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::internal::GatherCompressed>(op)->get_batch_dims());
        reverseIndexing = true;
    }

    if (batchDims < 0) {
        batchDims += indicesRank;
    }
    CPU_NODE_ASSERT(batchDims >= 0 && batchDims <= std::min(dataSrcRank, static_cast<int>(indicesRank)),
                    "has incorrect batch_dims ",
                    batchDims,
                    "!");

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0) {
            axis += dataSrcRank;
        }
        CPU_NODE_ASSERT(axis >= 0 && axis < dataSrcRank && batchDims <= axis,
                        "has incorrect input parameter axis value: ",
                        axis);
    }

    if (auto* indices = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_INDICES))) {
        constIndices = indices->cast_vector<int>();
    }
}

void Gather::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!fusedWith.empty()) {
        outPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    dataTypeSize = dataPrecision.size();
    outTypeSize = outPrecision.size();

    const auto& dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
    if (isAxisInputConst && isDataShapeStat) {
        axisDim = dataDims[axis];
        beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1LU, std::multiplies<>());
        betweenBatchAndAxisSize =
            std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1LU, std::multiplies<>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1LU, std::multiplies<>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        afterAxisSizeInBytesOut = afterAxisSize * outTypeSize;
        axisAndAfterAxisSize = axisDim * afterAxisSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSize = betweenBatchAndAxisSize * axisAndAfterAxisSize;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    }
    if (isDataShapeStat) {
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1LU, std::multiplies<>());
    }
    if (isIdxShapeStat) {
        const auto& idxDims = getInputShapeAtPort(GATHER_INDICES).getDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1LU, std::multiplies<>());

        if (isDataShapeStat) {
            specIdxAndAfterAxSize = specIndicesSize * afterAxisSize;
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            specIdxAndAfterAxSizeBOut = specIndicesSize * afterAxisSizeInBytesOut;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }
    if (compressed) {
        // gatherCompressed support input precision (u4/i4/u8/i8) to output precision (f16/bf16/f32).
        if (none_of(dataPrecision, ov::element::u8, ov::element::u4, ov::element::i8, ov::element::i4)) {
            dataPrecision = ov::element::f32;
        }

        ov::element::Type scalePrecision = getOriginalInputPrecisionAtPort(GATHER_SCALE);
        if (scalePrecision != ov::element::f32) {
            scalePrecision = ov::element::f32;
        }

        if (none_of(outPrecision, ov::element::f32, ov::element::f16, ov::element::bf16)) {
            outPrecision = ov::element::f32;
        }
        const size_t scale_count = getInputShapeAtPort(GATHER_SCALE).getElementsCount();
        CPU_NODE_ASSERT(scale_count != 0, "GATHER_SCALE input has zero elements (zero-dim shape)");
        scale_group_size =
            getInputShapeAtPort(GATHER_DATA).getElementsCount() / scale_count;
        have_scalar_scale = getInputShapeAtPort(GATHER_SCALE).getElementsCount() == 1U;

        if (getOriginalInputsNumber() == 5U) {
            ov::element::Type zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
            if (zpPrecision != ov::element::f32) {
                zpPrecision = ov::element::f32;
            }

            have_zp = true;
            have_scalar_zp = getInputShapeAtPort(GATHER_ZP).getElementsCount() == 1U;
            const size_t zp_count = getInputShapeAtPort(GATHER_ZP).getElementsCount();
            CPU_NODE_ASSERT(zp_count != 0, "GATHER_ZP input has zero elements (zero-dim shape)");
            zp_group_size =
                getInputShapeAtPort(GATHER_DATA).getElementsCount() / zp_count;
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                                  {LayoutType::ncsp, ov::element::i32},
                                  {LayoutType::ncsp, ov::element::i32},
                                  {LayoutType::ncsp, scalePrecision},
                                  {LayoutType::ncsp, zpPrecision}},
                                 {{LayoutType::ncsp, outPrecision}},
                                 ref_any);
        } else {
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                                  {LayoutType::ncsp, ov::element::i32},
                                  {LayoutType::ncsp, ov::element::i32},
                                  {LayoutType::ncsp, scalePrecision}},
                                 {{LayoutType::ncsp, outPrecision}},
                                 ref_any);
        }
        return;
    }  // Implementation desc type will be redefined in the fn prepareParams if a kernel will be created.
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32, isAxisInputConst}},
                         {{LayoutType::ncsp, outPrecision}},
                         ref_any);

    // Let's check for the special inPlace memory use case
    // in place only makes sense when we split by dense blocks since strided tensors are not supported by most nodes

    if (dataPrecision != outPrecision) {
        return;
    }

    if (!isAxisInputConst) {
        return;
    }

    if (batchDims != 0) {
        return;
    }

    if (constIndices.size() != 1) {
        return;
    }

    const auto& parentDims = inputShapes[0].getDims();
    const auto axisDim = parentDims[axis];
    if (Shape::UNDEFINED_DIM == axisDim) {
        return;
    }

    const auto indx = constIndices.front();
    const auto normIndex = indx < 0 ? static_cast<int64_t>(axisDim) + indx : indx;

    if (normIndex < 0 || normIndex >= static_cast<int64_t>(axisDim)) {
        return;
    }

    if (std::any_of(parentDims.begin(), parentDims.begin() + axis, [](size_t dim) {
            return dim != 1;
        })) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision, false, GATHER_DATA}},
                         unknown);
}

void Gather::createPrimitive() {
    if (isInPlace()) {
        return;
    }
    m_threads_num = parallel_get_max_threads();
#if defined(OPENVINO_ARCH_X86_64)
    uint64_t idxElPerVec = 1;
    if (!isDynamicNode()) {
        if (x64::mayiuse(x64::avx512_core)) {
            idxElPerVec = x64::cpu_isa_traits_t<x64::avx512_core>::vlen / idxTypeSize;
        } else if (x64::mayiuse(x64::avx2)) {
            idxElPerVec = x64::cpu_isa_traits_t<x64::avx2>::vlen / idxTypeSize;
        } else {
            idxElPerVec = 1;
        }
    }
    // Gather instruction is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_core) || x64::mayiuse(x64::avx2)) &&
        (isDynamicNode() || afterAxisSize == 1 ||
         (afterAxisSize <= idxElPerVec &&
          (x64::mayiuse(x64::avx512_core) || (x64::mayiuse(x64::avx2) && dataTypeSize == 4))))) {
        jGatherConfParams jcp;
        jcp.dataTypeSize = dataTypeSize;
        jcp.in_prec = dataPrecision;
        jcp.out_prec = outPrecision;
        jcp.reverseIndexing = reverseIndexing;
        jcp.dynamicShapes = isDynamicNode();
        jcp.batchDims = batchDims;
        if (