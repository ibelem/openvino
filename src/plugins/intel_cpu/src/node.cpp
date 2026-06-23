// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "edge.h"
#include "graph_context.h"
#include "itt.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/eltwise.h"
#include "nodes/input.h"
#include "nodes/node_config.h"
#include "nodes/reference.h"
#include "nodes/reorder.h"
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/cc/factory.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/util/pp.hpp"
#include "partitioned_mem_blk.h"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "utils/cpu_utils.hpp"
#endif
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

#ifndef CPU_DEBUG_CAPS
#    include <ostream>
#endif

using namespace dnnl;
using namespace openvino;
using namespace ov::intel_cpu::node;

namespace ov::intel_cpu {

Node::NodesFactory& Node::factory() {
    static NodesFactory factoryInstance;
    return factoryInstance;
}

Node::Node(const std::shared_ptr<ov::Node>& op, GraphContext::CPtr ctx, const ShapeInferFactory& shapeInferFactory)
    : context(std::move(ctx)),
      fusingPort(-1),
      engine(context->getEngine()),
      name(op->get_friendly_name()),
      typeStr(op->get_type_name()),
      type(TypeFromName(op->get_type_name())) {
    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto& shape = op->get_input_partial_shape(i);
        OPENVINO_ASSERT(!shape.rank().is_dynamic(),
                        "Unexpected: CPU plug-in doesn't support ",
                        getTypeStr(),
                        " operation with dynamic rank. Operation name: ",
                        getName());

        bool isScalar = shape.rank().get_length() == 0;
        inputShapes.emplace_back(isScalar ? ov::PartialShape{1} : shape);
        originalInputPrecisions.emplace_back(op->get_input_element_type(i));
    }

    parentEdges.reserve(inputShapes.size());

    if (typeStr != "Result" && typeStr != "Assign") {
        OPENVINO_ASSERT(op->get_output_size() != 0,
                        "Node with type '",
                        typeStr,
                        "' and name '",
                        name,
                        "' does not have any outputs.");
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto& shape = op->get_output_partial_shape(i);
            OPENVINO_ASSERT(!shape.rank().is_dynamic(),
                            "Unexpected: CPU plug-in doesn't support ",
                            getTypeStr(),
                            " operation with dynamic rank. Operation name: ",
                            getName());

            bool isScalar = shape.rank().get_length() == 0;
            outputShapes.emplace_back(isScalar ? ov::PartialShape{1} : shape);
            originalOutputPrecisions.emplace_back(op->get_output_element_type(i));
        }

        childEdges.reserve(outputShapes.size());
    }

    isDynamic = std::any_of(inputShapes.begin(),
                            inputShapes.end(),
                            [](const Shape& shape) {
                                return shape.isDynamic();
                            }) ||
                std::any_of(outputShapes.begin(), outputShapes.end(), [](const Shape& shape) {
                    return shape.isDynamic();
                });

    if (isDynamic) {
        shapeInference = shapeInferFactory.makeShapeInfer();
    }

    const auto& rtInfo = op->get_rt_info();
    originalLayers = getRTInfoValue(rtInfo, "originalLayersNames");

    if (originalLayers.empty()) {
        addOriginalLayer(name);
    }

    const auto& primitivesPriority = getImplPriorityValue(op);
    if (!primitivesPriority.empty()) {
        std::istringstream stream(primitivesPriority);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            customImplPriorities.push_back(parse_impl_name(str));
            OPENVINO_ASSERT(customImplPriorities.back() != impl_desc_type::unknown || str == "cpu:unknown",
                            "Unsupported CPU implementation ",
                            str,
                            " for node ",
                            getName());
        }
        const auto& defaultImplPriorities = getDefaultImplPriority();
        customImplPriorities.insert(customImplPriorities.end(),
                                    defaultImplPriorities.begin(),
                                    defaultImplPriorities.end());
    }

    std::string inputMemoryFormats = getInputMemoryFormats(op);
    if (!inputMemoryFormats.empty()) {
        std::istringstream stream(inputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            memoryFormatFilter.input.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    std::string outputMemoryFormats = getOutputMemoryFormats(op);
    if (!outputMemoryFormats.empty()) {
        std::istringstream stream(outputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            memoryFormatFilter.output.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    const auto it = rtInfo.find("enforceBF16evenForGraphTail");
    if (it != rtInfo.end()) {
        enforceBF16evenForGraphTail = it->second.as<bool>();
    }
    if (is_conversion_disabled(op, element::f16)) {
        keepOriginalPrecision = true;
    }
}

Node::Node(const std::string& type,
           std::vector<Shape> inShapes,
           std::vector<Shape> outShapes,
           std::vector<ov::element::Type> inputPrecisions,
           std::vector<ov::element::Type> outputPrecisions,
           std::string name,
           const GraphContext::CPtr& ctx)
    : inputShapes(std::move(inShapes)),
      outputShapes(std::move(outShapes)),

      context(ctx),
      originalInputPrecisions(std::move(inputPrecisions)),
      originalOutputPrecisions(std::move(outputPrecisions)),
      fusingPort(-1),
      engine(ctx->getEngine()),
      name(std::move(name)),
      typeStr(type),
      type(TypeFromName(type)) {
    parentEdges.reserve(inputShapes.size());
    childEdges.reserve(outputShapes.size());
}

void Node::addEdge(const EdgePtr& edge) {
    auto parent = edge->getParent();
    auto child = edge->getChild();
    assert(parent && child);

    parent->addChildEdge(edge);
    child->addParentEdge(edge);
}

void Node::remove() {
    auto drop = [](const std::vector<EdgeWeakPtr>& edges) {
        for (const auto& edge : edges) {
            auto edgePtr = edge.lock();
            if (!edgePtr) {
                continue;
            }
            edgePtr->getParent()->removeChildEdge(edgePtr);
            edgePtr->getChild()->removeParentEdge(edgePtr);
        }
    };

    drop(parentEdges);
    drop(childEdges);
}

bool Node::isEdgesEmpty(const std::vector<EdgeWeakPtr>& edges) {
    return std::all_of(edges.begin(), edges.end(), [](const EdgeWeakPtr& edge) {
        return !edge.lock();
    });
}

void Node::createPrimitive() {
    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void Node::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), false);
}

void Node::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs) {
    for (const auto& type : priority) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            const auto& supportedPrimitiveDesc = getSupportedPrimitiveDescriptors()[i];
            const impl_desc_type supportedType = supportedPrimitiveDesc.getImplementationType();

            if (supportedType != type) {
                continue;
            }

            int equalsLocalFormatCount = 0;
            const size_t descInConfSize = supportedPrimitiveDesc.getConfig().inConfs.size();

            OPENVINO_ASSERT(descInConfSize <= getParentEdges().size(),
                            getName(),
                            " Desc ",
                            i,
                            " with type: ",
                            supportedType,
                            " has more input ports than node: ",
                            descInConfSize,
                            " vs ",
                            getParentEdges().size());

            for (size_t j = 0; j < descInConfSize; j++) {
                auto parentEdge = getParentEdgeAt(j);
                auto parentPtr = parentEdge->getParent();

                // We don't take into account constant edges since reorders on them will be executed on load network
                // stage
                if (ignoreConstInputs && j > 0 && parentPtr->isConstant()) {
                    equalsLocalFormatCount++;
                    continue;
                }

                auto* parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                    int inNum = parentEdge->getInputNum();
                    if (inNum < 0 || inNum >= static_cast<int>(parent_spd->getConfig().outConfs.size())) {
                        inNum = 0;
                    }
                    auto curDesc = supportedPrimitiveDesc.getConfig().inConfs[j].getMemDesc();
                    auto parentDesc = parent_spd->getConfig().outConfs[