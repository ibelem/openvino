// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool ReorgYolo::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto reorgYolo = ov::as_type_ptr<const ov::op::v0::ReorgYolo>(op);
        if (!reorgYolo) {
            errorMessage = "Only v0 ReorgYolo operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ReorgYolo::ReorgYolo(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    CPU_NODE_ASSERT(all_of(1U, getOriginalInputsNumber(), getOriginalOutputsNumber()),
                    "has incorrect number of input/output edges!");

    const auto reorgYolo = ov::as_type_ptr<const ov::op::v0::ReorgYolo>(op);
    const auto strides = reorgYolo->get_strides();
    CPU_NODE_ASSERT(!strides.empty(), "has empty strides");
    stride = strides[0];
}

void ReorgYolo::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ReorgYolo::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void ReorgYolo::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto* src_data = getSrcDataAtPortAs<const float>(0);
    auto* dst_data = getDstDataAtPortAs<float>(0);

    const auto& inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    size_t IW = (inDims.size() > 3) ? inDims[3] : 1;
    size_t IH = (inDims.size() > 2) ? inDims[2] : 1;
    size_t IC = (inDims.size() > 1) ? inDims[1] : 1;
    size_t B = (!inDims.empty()) ? inDims[0] : 1;

    OPENVINO_ASSERT(B * IC * IH * IW <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                    "ReorgYolo: tensor size exceeds the supported range");

    size_t ic_off = IC / (static_cast<size_t>(stride) * static_cast<size_t>(stride));
    size_t ih_off = IH * static_cast<size_t>(stride);
    size_t iw_off = IW * static_cast<size_t>(stride);
    for (size_t b = 0; b < B; b++) {
        for (size_t ic = 0; ic < IC; ic++) {
            for (size_t ih = 0; ih < IH; ih++) {
                for (size_t iw = 0; iw < IW; iw++) {
                    size_t dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;

                    size_t oc = ic % ic_off;
                    size_t offset = ic / ic_off;

                    size_t ow = iw * static_cast<size_t>(stride) + offset % static_cast<size_t>(stride);
                    size_t oh = ih * static_cast<size_t>(stride) + offset / static_cast<size_t>(stride);

                    size_t srcIndex = b * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;

                    dst_data[dstIndex] = src_data[srcIndex];
                }
            }
        }
    }
}

bool ReorgYolo::created() const {
    return getType() == Type::ReorgYolo;
}

}  // namespace ov::intel_cpu::node