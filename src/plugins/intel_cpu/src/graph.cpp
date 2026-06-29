// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.h"

#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "allocation_context.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "edge.h"
#include "graph_context.h"
#include "graph_dumper.h"
#include "graph_optimizer.h"
#include "infer_request.h"
#include "itt.h"
#include "memory_control.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_state.h"
#include "node.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/convert.h"
#include "nodes/input.h"
#include "nodes/memory.hpp"
#include "nodes/reorder.h"
#include "nodes/subgraph.h"
#include "nodes/tensoriterator.h"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "perf_count.h"
#include "proxy_mem_blk.h"
#include "thread_pool_imp.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/node_dumper.h"
#include "utils/verbose.h"
#include "weights_cache.hpp"
#ifdef CPU_DEBUG_CAPS
#    include "openvino/core/partial_shape.hpp"
#endif

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE || \
     OV_THREAD == OV_THREAD_OMP)
#    include <atomic>
#endif

#if OV_THREAD_USE_TBB
#    include <tbb/task.h>
#endif

#if defined(OPENVINO_ARCH_X86_64) && defined(__linux__)
#    include "openvino/runtime/properties.hpp"
#endif

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    include <common/primitive_desc_iface.hpp>

#    include "onednn/iml_type_mapper.h"
#    include "utils/precision_support.h"
#endif

using namespace dnnl;

namespace ov::intel_cpu {

Graph::~Graph() {
    CPU_DEBUG_CAP_ENABLE(summary_perf(*this));
    CPU_DEBUG_CAP_ENABLE(average_counters(*this));
    CPU_DEBUG_CAP_ENABLE(serialize(*this));
}

template <typename NET>
void Graph::CreateGraph(NET& model, const GraphContext::CPtr& context) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "CreateGraph");

    Init(model, context);

    Activate();
}

void Graph::Init(const std::vector<NodePtr>& graphNodes,
                 const std::vector<EdgePtr>& graphEdges,
                 const GraphContext::CPtr& context,
                 std::string name) {
    if (IsReady()) {
        ForgetGraphData();
    }

    m_context = context;
    m_stream = make_stream(getEngine(), m_context->getCpuParallel()->get_thread_pool());
    m_context->getCpuParallel()->activate();

    this->_name = std::move(name);

    this->graphNodes = graphNodes;
    this->graphEdges = graphEdges;

    for (const auto& node : graphNodes) {
        if ("Parameter" == node->getTypeStr()) {
            inputNodes.push_back(node);
        } else if ("Result" == node->getTypeStr()) {
            outputNodes.push_back(node);
        }
    }

    Configure();
}

void Graph::CreateGraph(const std::vector<NodePtr>& graphNodes,
                        const std::vector<EdgePtr>& graphEdges,
                        const GraphContext::CPtr& context,
                        std::string name) {
    Init(graphNodes, graphEdges, context, std::move(name));

    Activate();
}

template void Graph::CreateGraph(const std::shared_ptr<const ov::Model>&, const GraphContext::CPtr&);

void Graph::Replicate(const std::shared_ptr<const ov::Model>& model,
                      const std::vector<node::Input::InputConfig>& inputConfigs,
                      const std::vector<node::Input::OutputConfig>& outputConfigs) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::ov_intel_cpu_LT, "Graph::Replicate", "ov::Model");

    this->_name = model->get_friendly_name();

    // Map data object onto producer node
    std::map<std::shared_ptr<ov::Node>, NodePtr> op2node;

    // nodes which has no consumers (output or just unused). But doesn't marked as graph output.
    // Will be stored as fake output separately.
    std::deque<ov::Output<ov::Node>> unusedOutputs;

    auto getParentOutputPort = [](const std::shared_ptr<ov::Node>& childOp,
                                  const std::shared_ptr<ov::Node>& parentOp,
                                  const size_t childInputPort) -> int {
        for (size_t parentPort = 0; parentPort < parentOp->get_output_size(); parentPort++) {
            if (childOp->input(childInputPort).get_tensor_ptr() == parentOp->output(parentPort).get_tensor_ptr()) {
                return static_cast<int>(parentPort);
            }
        }

        return -1;
    };

    auto createNode = [&](const std::shared_ptr<ov::Node>& op) -> NodePtr {
        // special handling for Parameters and Results
        if (op->get_type_info() == op::v0::Parameter::get_type_info_static()) {
            auto input_index = model->get_parameter_index(ov::as_type_ptr<op::v0::Parameter>(op));
            OPENVINO_ASSERT(input_index >= 0,
                            "CPU plugin cannot find op: ",
                            op->get_friendly_name(),
                            " in model parameter list!");

            const auto& config = static_cast<size_t>(input_index) < inputConfigs.size() ? inputConfigs[input_index]
                                                                                        : node::Input::InputConfig{};
            NodePtr node = std::make_shared<node::Input>(op, m_context, config);
            inputNodes[input_index] = node;

            if (node->isDynamicNode()) {
                graphHasDynamicInput = true;
            }

            return node;
        }

        if (op->get_type_info() == op::v0::Result::get_type_info_static()) {
            auto output_index = model->get_result_index(ov::as_type_ptr<op::v0::Result>(op));
            OPENVINO_ASSERT(output_index >= 0,
                            "CPU plugin cannot find op: ",
                            op->get_friendly_name(),
                            " in model result list!");

            const auto& config = static_cast<size_t>(output_index) < outputConfigs.size() ? outputConfigs[output_index]
                                                                                          : node::Input::OutputConfig{};
            NodePtr node = std::make_shared<node::Input>(op, m_context, config);
            outputNodes[output_index] = node;

            return node;
        }

        return NodePtr(Node::factory().create(op, m_context));
    };

    inputNodes.resize(model->get_parameters().size());
    outputNodes.resize(model->get_results().size());

    for (const auto& op : model->get_ordered_ops()) {
        const NodePtr node = createNode(op);

        AddNode(node);
        op2node[op] = node;

        for (size_t port = 0; port < op->get_input_size(); port++) {
            auto parentOp = op->get_input_node_shared_ptr(port);
            auto parentNode = op2node[parentOp];

            CreateEdge(parentNode, node, getParentOutputPort(op, parentOp, port), static_cast<int>(port));
        }

        if (none_of(op->get_type_info(),
                    op::v0::Result::get_type_info_static(),
                    op::v3::Assign::get_type_info_static(),
                    op::v6::Assign::get_type_info_static())) {
            for (size_t oi = 0; oi < op->get_output_size(); oi++) {
                if (op->get_output_target_inputs(oi).empty()) {
                    unusedOutputs.push_back(op->output(oi));
                }
            }
        }
    }

    // Add stub output node for unused data
    for (const auto& unusedOutput : unusedOutputs) {
        auto parentNode = op2node[unusedOutput.get_node_shared_ptr()];
        const auto port = unusedOutput.get_index();
        const auto nodeName =
            std::string("stub_") + std::to_string(unusedOutput.get_index()) + "_" + parentNode->getName();
        const NodePtr outNode = std::make_shared<node::Input>(parentNode->outputShapes[port],
                                                              parentNode->getOriginalOutputPrecisionAtPort(port),
                                                              nodeName,
                                                              "Result",
                                                              m_context);
        CreateEdge(parentNode, outNode, port, 0);
        AddNode(outNode);
    }

    auto hasSubgraphConsumers = [](const NodePtr& node) -> bool {
        const auto& childEdges = node->getChildEdges();
        return std::any_of(childEdges.begin(), childEdges.end(), [](const EdgeWeakPtr& edge) -> bool {
            auto edgePtr = edge.lock();
            if (!edgePtr) {
                return false;
            }
            return edgePtr->getChild()->getType() == Type::Subgraph;
        });
    };

    // enforce must be performed after inputs and outputs info are taken into account
    EnforceInferencePrecision();

    // update input precisions of consumers to avoid extra reorders
    for (auto& inputNode : inputNodes) {
        const auto precToSet = inputNode->getOriginalOutputPrecisionAtPort(0);
        const auto childEdges = inputNode->getChildEdgesAtPort(0);
        for (const auto& childEdge : childEdges) {
            const auto child = childEdge->getChild();
            const auto child_prec = child->getOriginalInputPrecisionAtPort(childEdge->getOutputNum());
            if (none_of(child_prec, ov::element::bf16, ov::element::f16) &&
                // remove this WA when #78939 is resolved
                !hasSubgraphConsumers(child)) {
                child->setOriginalInputPrecisionAtPort(childEdge->getOutputNum(), precToSet);
            }
        }
    }

    // update output precisions of producers to avoid extra reorders
    // do this only in case output configuration is not provided explicitly
    if (outputConfigs.empty()) {
        for (auto& outputNode : outputNodes) {
            const auto precToSet = outputNode->getOriginalInputPrecisionAtPort(0);
            const auto parentEdge = outputNode->getParentEdgeAt(0);
            const auto parent = parentEdge->getParent();
            parent->setOriginalOutputPrecisionAtPort(parentEdge->getInputNum(), precToSet);
        }
    }
}

static std::vector<size_t> IdentifySyncPoints(const std::vector<NodePtr>& graphNodes) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::IdentifySyncPoints");
    std::vector<size_t> syncNodesInds;

    for (size_t i = 0; i < graphNodes.size(); ++i) {
        const auto& node = graphNodes[i];

        if (!node->isDynamicNode()) {
            continue;
        }

        if (node->outputShapeDataDependency() ||
            // WA: for convolution plus sum(broadcast). Due to the fact that a convolution with sum use the same memory
            // for second sum term and the output tensors (inPlace) resizing the output tensor, may lead to reallocation
            // of this second term memory and possible data lost. The reallocation may happen when the second term shape
            // is broadcasted to the output tensor shape. To avoid the data loss, we have a special processing for such
            // cases inside the convolution node, but it works properly only when dynamic shapes inference, preparation
            // and execution a called for this node sequentially.
            (node->getType() == Type::Convolution && node->isInPlace()) ||
            // Due to the special handling of the internal states and initialization subgraphs, MemoryInput nodes must
            // be processed as a internal dynamism node, allowing to hide the aforementioned complexity inside the
            // MemoryInput::executeDynamic implementation
            (node->getType() == Type::MemoryInput)) {
            syncNodesInds.push_back(i);
        }
    }

    return syncNodesInds;
}

static std::tuple<std::vector<NodePtr>, std::vector<size_t>> ExtractExecutableNodesAndSyncPoints(
    const std::vector<size_t>& syncNodesInds,
    const std::vector<NodePtr>& graphNodes) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::ExtractExecutableNodesAndSyncPoints");
    std::unordered_map<size_t, size_t> graphIdToExecutableId;
    std::vector<NodePtr> executableGraphNodes;
    for (size_t i = 0; i < graphNodes.size(); i++) {
        const auto& node = graphNodes[i];
        const bool staticZeroDims = !node->isDynamicNode() && !node->isExecutable() && !node->isInPlace();
        const bool dynamicNonInputOutput = node->isDynamicNode() && none_of(node->getType(), Type::Input, Type::Output);

        if (!node->isConstant() &&  // constants are executed once in scope of compile_model
            !staticZeroDims &&      // never execute static nodes with zero dim input / output tensors
            (CPU_DEBUG_CAPS_ALWAYS_TRUE(!node->neverExecute()) ||  // execute all executable nodes
             dynamicNonInputOutput)) {                             // plus dynamic ones, except inputs / outputs
            graphIdToExecutableId[i] = executableGraphNodes.size();
            executableGraphNodes.emplace_back(node);
        }
    }

    // use set to ensure sorted unique sync entries
    std::set<size_t> uniqueExecutableSyncNodesInds;
    for (const auto& syncNodesInd : syncNodesInds) {
        auto it = graphIdToExecutableId.find(syncNodesInd);
        if (it != graphIdToExecutableId.end()) {
            uniqueExecutableSyncNodesInds.insert(it->second);
            // since sometimes we need to run the synchronization node  alone (for example in the case of internal
            // dynamism) let's add another sync index after the sync point node
            uniqueExecutableSyncNodesInds.insert(it->second + 1);
        }
    }
    uniqueExecutableSyncNodesInds.insert(executableGraphNodes.size());
    // convert to a vector to reduce runtime overhead
    std::vector<size_t> executableSyncNodesInds(uniqueExecutableSyncNodesInds.begin(),
                                                uniqueExecutableSyncNodesInds.end());

    return std::make_tuple(std::move(executableGraphNodes), std::move(executableSyncNodesInds));
}

void Graph::Init(const std::shared_ptr<const ov::Model>& model,
                 const GraphContext::CPtr& context,
                 const std::vector<node::Input::InputConfig>& inputConfigs,
                 const std::vector<node::Input::OutputConfig>& outputConfigs) {
    if (IsReady()) {
        ForgetGraphData();
    }

    m_context = context;
    m_stream = make_stream(getEngine(), m_context->getCpuParallel()->get_thread_pool());
    m_context->getCpuParallel()->activate();

    Replicate(model, inputConfigs, outputConfigs);

    Configure();
}

void Graph::Activate() {
    // @todo It is possible that execution graph is already created in scope of
    // the allocation context collection from the outer graph so the state for inner graph is "Ready"
    // We probably want to avoid such uncertainty
    // OPENVINO_ASSERT(status == Status::Initialized, "Invalid graph status: ", static_cast<int>(status));
    Allocate();

    CreatePrimitivesAndExecConstants();

#ifndef CPU_DEBUG_CAPS
    for (auto& graphNode : graphNodes) {
        graphNode->cleanup();
    }
#endif

    CPU_DEBUG_CAP_ENABLE(serialize(*this));
}

void Graph::Configure([[maybe_unused]] bool optimize) {
    OPENVINO_ASSERT(status == Status::NotReady, "Invalid graph status");

    SortTopologically();
    InitNodes();

    ov::intel_cpu::GraphOptimizer::ApplyCommonGraphOptimizations(*this);

    SortTopologically();

    InitDescriptors();

    ResolveInplaceDirections();

    InitOptimalPrimitiveDescriptors();

    ResolveEdgeConflicts();

    ov::intel_cpu::GraphOptimizer::ShareReorders(*this);
    RemoveDroppedNodes();

    SortTopologically();

    ResolveComplexInplaceConflicts();

    ov::intel_cpu::GraphOptimizer::ApplyImplSpecificGraphOptimizations(*this);

    SortTopologically();

    ResolveComplexInplaceConflicts();

    SortTopologically();

    status = Status::Initialized;
}

void Graph::InitNodes() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::InitNodes");
    for (auto& node : graphNodes) {
        node->init();
    }
}

void Graph::InitDescriptors() {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::ov_intel_cpu_LT, "InitDescriptors", "Prepare");

    for (auto& node : graphNodes) {
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.getSupportedDescriptors);
        DEBUG_LOG("Get supported primitive descriptors for node: ", node->getName());
        node->getSupportedDescriptors();

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.initSupportedPrimitiveDescriptors);
        DEBUG_LOG("Init supported primitive descriptors for node: ", node->getName());
        node->initSupportedPrimitiveDescriptors();
#ifdef CPU_DEBUG_CAPS
        {
            const auto& SPDs = node->getSupportedPrimitiveDescriptors();
            for (size_t i = 0; i < SPDs.size(); i++) {
                DEBUG_LOG("#",
                          node->getExecIndex(),
                          " ",
                          node->getName(),
                          " Before filter, SupportedPrimitiveDescriptors [",
                          i,
                          "/",
                          SPDs.size(),
                          "]: \n",
                          SPDs[i]);
            }
        }
#endif
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.filterSupportedPrimitiveDescriptors);
        DEBUG_LOG("Filter supported primitive descriptors for node: ", node->getName());
        node->filterSupportedPrimitiveDescriptors();

#ifdef CPU_DEBUG_CAPS
        const auto& SPDs = node->getSupportedPrimitiveDescriptors();
        for (size_t i = 0; i < SPDs.size(); i++) {
            DEBUG_LOG("#",
                      node->getExecIndex(),
                      " ",
                      node->getName(),
                      " After filter,  SupportedPrimitiveDescriptors [",
                      i,
                      "/",
                      SPDs.size(),
                      "]: \n",
                      SPDs[i]);
        }
#endif
    }

    for (auto& node : graphNodes) {
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.selectOptimalPrimitiveDescriptor);
        DEBUG_LOG("Select optimal primitive descriptors for node: ", node->getName());
        node->selectOptimalPrimitiveDescriptor();
    }
}

void Graph::ResolveInplaceDirections() {
    OV_ITT_SCOPED_TASK(itt::domains::ov_intel_cpu, "Graph::ResolveInplaceDirections");

    for (auto& node : graphNodes) {
        node->resolveInPlaceDirection();
    }
}

void Graph::InitOptimalPrimitiveDescriptors() {
    OV_ITT_SCOPED_TASK(itt::domains::ov_intel_cpu, "Graph::InitOptimalPrimitiveDescriptors");
    for (auto& node : graphNodes) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, node->profiling.initOptimalPrimitiveDescriptor);
        DEBUG_LOG("Init optimal primitive descriptors for node: ", node->getName());
        node->initOptimalPrimitiveDescriptor();
        DEBUG_LOG("#",
                  node->getExecIndex(),
                  " ",
                  node->getName(),
                  "\n",
                  *node->getSelectedPrimitiveDescriptor(),
                  "selectedPrimitiveDescriptorIdx = ",
                  node->selectedPrimitiveDescriptorIndex);
    }
}

void Graph::CreatePrimitivesAndExecConstants() const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::CreatePrimitivesAndExecConstants");
    using shared_memory_ptr = WeightsSharing::SharedMemory::Ptr;

    auto acquireSharedOutputs = [this](const NodePtr& node) {
        std::vector<shared_memory_ptr> outputs;
        bool hasLocalAllocatedEdges = false;
        bool hasExternalInvalidEdges = false;

        for (size_t i = 0; i < node->getChildEdges().size(); ++i) {
            auto edgePtr = node->getChildEdgeAt(i);
            if (edgePtr) {
                if (edgePtr->isUseExternalMemory()) {
                    auto ptr = m_context->getWeightsCache()->get(edgePtr->hash());
                    outputs.emplace_back(ptr);
                    if (!ptr->isValid()) {
                        hasExternalInvalidEdges = true;
                    }
                } else {
                    hasLocalAllocatedEdges = true;
                }
            }
        }

        return std::make_tuple(hasExternalInvalidEdges, hasLocalAllocatedEdges, outputs);
    };

    for (const auto& node : graphNodes) {
        {
            OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, node->profiling.createPrimitive);
            DEBUG_LOG(*node);
            node->createPrimitive();
        }

        if (!node->isConstant() || !node->isExecutable()) {
            continue;
        }

        if (m_context->getWeightsCache()) {
            auto sharedOutputs = acquireSharedOutputs(node);

            if (std::get<0>(sharedOutputs) || std::get<1>(sharedOutputs)) {
                ExecuteNodeWithCatch(node);

                for (auto& output : std::get<2>(sharedOutputs)) {
                    output->valid(true);
                }
            }
        } else {
            ExecuteNodeWithCatch(node);
        }
    }
}

static bool isReorderAvailable(const MemoryDescPtr& parentDesc,
                               const MemoryDescPtr& childDesc,
                               const dnnl::engine& eng) {
    auto definedParentDesc = parentDesc->isDefined() ? parentDesc : MemoryDescUtils::makeDummyDesc(*parentDesc);
    memory::desc srcMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(definedParentDesc)->getDnnlDesc();

    auto definedChildDesc = childDesc->isDefined() ? childDesc : MemoryDescUtils::makeDummyDesc(*childDesc);
    memory::desc dstMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(definedChildDesc)->getDnnlDesc();

    dnnl::primitive_attr attr;

    dnnl_primitive_desc_t result = nullptr;
    auto status = dnnl_reorder_primitive_desc_create(&result,
                                                     srcMemDesc.get(),
                                                     eng.get(),
                                                     dstMemDesc.get(),
                                                     eng.get(),
                                                     attr.get());
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    // temporary WA for slow FP32->FP16 conversion reorder in oneDNN on ARM
    // pretend the reorder is not available to use Convert node instead
    if (hasHardwareSupport(ov::element::f16) && (result != nullptr) &&
        parse_impl_name(result->impl()->name()) == ref_any) {
        dnnl_primitive_desc_destroy(result);
        return false;
    }
#endif
    if (result) {
        dnnl_primitive_desc_destroy(result);
    }

    return dnnl_success == status;
}

void Graph::insertReorder(EdgePtr& edge, bool isOptimized, std::unordered_set<std::string>& uniqueLayerNames) {
    std::string basicLayerName = edge->getParent()->getName() + "_" +
                                 node::Reorder::getReorderArgs(edge->getInputDesc(), edge->getOutputDesc()) + "_" +
                                 edge->getChild()->getName();
    std::string layerName = basicLayerName;
    int idx = 0;
    while (uniqueLayerNames.find(layerName) != uniqueLayerNames.end()) {
        idx++;
        layerName = basicLayerName + "_" + std::to_string(idx);
    }
    uniqueLayerNames.insert(layerName);

    // optimized flag indicate that just desc update w/o actual physical memory movement.
    InsertReorder(edge, layerName, edge->getInputDesc(), edge->getOutputDesc(), isOptimized);
}

void Graph::insertConvert(EdgePtr& edge) {
    const auto& inDesc = edge->getInputDesc();
    const auto& outDesc = edge->getOutputDesc();

    std::string convertName = edge->getParent()->getName() + "_" + inDesc.getPrecision().get_type_name() + "_" +
                              outDesc.getPrecision().get_type_name();

    auto convertNode = std::make_shared<node::Convert>(inDesc.getShape(),
                                                       inDesc.getPrecision(),
                                                       outDesc.getPrecision(),
                                                       convertName,
                                                       m_context);
    convertNode->setDescs(inDesc, outDesc);
    InsertNode(edge, convertNode, true);
}

static std::unordered_set<std::string> getUniqueLayerNames(const std::vector<NodePtr>& graphNodes) {
    std::unordered_set<std::string> uniqueLayerNames;
    uniqueLayerNames.reserve(graphNodes.size());

    for (const auto& node : graphNodes) {
        uniqueLayerNames.insert(node->getName());
    }

    return uniqueLayerNames;
}

void Graph::ResolveEdgeConflicts() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::ResolveEdgeConflicts");

    std::unordered_set<std::string> uniqueLayerNames = getUniqueLayerNames(graphNodes);

    /* When inserting convert / reorder, two new edges are added (pushed to the end) to the graphEdges.
       So use a plain for loop, to handle newly inserted edges as well */
    for (size_t i = 0; i < graphEdges.size(); i++) {  // NOLINT(modernize-loop-convert)
        auto& edge = graphEdges[i];
        auto reorderStatus = edge->needReorder();  // NOLINT(modernize-loop-convert)
        DEBUG_LOG(*edge, " reorderStatus = ", reorderStatus);

        switch (reorderStatus) {
        case Edge::ReorderStatus::Regular: {
            if (reorderStatus == Edge::ReorderStatus::Regular &&
                edge->getInputDesc().getPrecision() != edge->getOutputDesc().getPrecision() &&
                !isReorderAvailable(edge->getInputPortDesc()->getMemDesc(),
                                    edge->getOutputPortDesc()->getMemDesc(),
                                    this->getEngine())) {
                // just insert convert. If layout reorder is still needed, it will be inserted later in the traverse
                insertConvert(edge);
            } else {
                insertReorder(edge, false, uniqueLayerNames);
            }
            break;
        }
        case Edge::ReorderStatus::Optimized:
            insertReorder(edge, true, uniqueLayerNames);
            break;
        case Edge::ReorderStatus::No:
            break;
        }
    }

    RemoveDroppedEdges();
}

void Graph::ResolveComplexInplaceConflicts() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::ResolveComplexInplaceConflicts");

    auto numberOfEdges = static_cast<ptrdiff_t>(graphEdges.size());

    std::unordered_set<std::string> uniqueLayerNames = getUniqueLayerNames(graphNodes);

    auto updateEdge = [&](ptrdiff_t& i) {
        graphEdges.erase(graphEdges.begin() + i);
        i--;
        numberOfEdges--;
    };

    // secondary pass to eliminate complex inplace conflicts
    auto needReorder = [](const EdgePtr& edge) -> bool {
        int inNumber = edge->getInputNum();
        const auto portChildEdges = edge->getParent()->getChildEdgesAtPort(inNumber);
        if (portChildEdges.size() > 1) {
            if (auto modifyingNode = edge->modifiedInPlace()) {
                auto execIndex = modifyingNode->getExecIndex();
                for (const auto& pEdgePeer : portChildEdges) {
                    if (pEdgePeer == edge) {
                        continue;
                    }
                    std::vector<NodePtr> vecConsumers;
                    pEdgePeer->collectConsumers(vecConsumers);

                    for (const auto& node : vecConsumers) {
                        if (node->getExecIndex() >= execIndex ||
                            any_of(node->getType(), Type::MemoryOutput, Type::Output)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    };

    for (ptrdiff_t i = 0; i < numberOfEdges; i++) {
        auto edge = graphEdges[i];
        if (needReorder(edge)) {
            insertReorder(edge, false, uniqueLayerNames);
            updateEdge(i);
        }
    }
}

/**
 * Partition the \clusters of Edges, by moving to the end and allocating at the same time
 * the clusters that cannot be handled as part of the generic memory solver algorithm.
 * Such clusters meet one of the following criteria:
 * - base edge of a cluster is a "ov::element::string" type of edge
 * - base edge of a cluster is a Constant edge
 *
 * @return a remaining number of clusters to process (left partition)
 */
static size_t AllocateStringsAndConstants(EdgeClusters& clusters, size_t remaining, const GraphContext::CPtr& context) {
    auto allocateConstantEdge = [&context](const EdgePtr& edge) {
        if (edge->getParent()->getType() == Type::Input) {
            auto constNode = std::static_pointer_cast<node::Input>(edge->getParent());
            edge->reuse(std::const_pointer_cast<IMemory>(constNode->getMemoryPtr()));
        } else {
            edge->externalAllocate(context->getWeightsCache());
        }
    };

    auto allocateStringMemory = [&context](const EdgePtr& edge) {
        auto memory = std::make_shared<StringMemory>(context->getEngine(), edge->getOriginalDesc());
        edge->reuse(memory);
        return memory->getStringMemoryBlockPtr();
    };

    auto notAllocatedPartitionEnd = std::partition(
        clusters.begin(),
        clusters.begin() + remaining,
        [&allocateStringMemory, &allocateConstantEdge, &context](const EdgeCluster& cluster) {
            const auto& baseEdge = cluster.at(0);

            // Allocate a cluster of the constants
            if (baseEdge->getParent()->isConstant()) {
                allocateConstantEdge(baseEdge);
                return false;
            }

            // Allocate a non-constant string cluster
            if (baseEdge->getOriginalDesc().getPrecision() == element::string) {
                OPENVINO_ASSERT(std::all_of(cluster.begin(),
                                            cluster.end(),
                                            [](const EdgePtr& edge) {
                                                return edge->getOriginalDesc().getPrecision() == element::string;
                                            }),
                                "All edges in the string cluster must be strings.");
                auto memBlock = allocateStringMemory(baseEdge);
                // reuse starting from second edge (skip the base edge)
                for (size_t i = 1; i < cluster.size(); i++) {
                    const auto& edge = cluster.at(i);
                    edge->reuse(
                        std::make_shared<StringMemory>(context->getEngine(), edge->getOriginalDesc(), memBlock));
                }
                return false;
            }

            return true;
        });

    return std::distance(clusters.begin(), notAllocatedPartitionEnd);
}

/**
 * Partition the \clusters of Edges, by moving to the end and allocating at the same time
 * the clusters that have dynamic output edge as a base edge.
 * Also collect the memory blocks associated with dynamic output nodes, allowing infer requests to access them.
 *
 * @return a tuple of remaining number of clusters to process (left partition) and the output memory blocks
 */
static std::tuple<size_t, Graph::OutputMemoryBlocks>
AllocateDynamicOutputEdges(EdgeClusters& clusters, size_t remaining, const std::vector<NodePtr>& outputNodes) {
    Graph::OutputMemoryBlocks outputMemBlocks;

    auto collectDynamicOutputMemBlocks = [&outputMemBlocks, &outputNodes](const EdgeCluster& cluster) {
        const auto& baseEdge = cluster.at(0);
        const auto& desc = baseEdge->getOriginalDesc();
        const auto& child = baseEdge->getChild();
        const bool dynamicOutputEdge = child->getType() == Type::Output && !desc.isDefined();

        if (!dynamicOutputEdge) {
            return true;
        }

        auto proxyMemBlock = std::make_shared<ProxyMemoryBlock>();
        DEBUG_LOG("ProxyMemoryBlock ", proxyMemBlock);

        baseEdge->allocate(proxyMemBlock);

        int count = 0;
        for (size_t output_index = 0; output_index < outputNodes.size(); ++output_index) {
            if (outputNodes[output_index] == child) {
                outputMemBlocks[output_index] = proxyMemBlock;
                count++;
            }
        }
        // sometimes there are unused output ports.
        OPENVINO_ASSERT(count <= 1, "CPU plugin cannot find output node. count ", count);
        return false;
    };

    auto notAllocatedPartitionEnd =
        std::partition(clusters.begin(), clusters.begin() + remaining, collectDynamicOutputMemBlocks);

    remaining = std::distance(clusters.begin(), notAllocatedPartitionEnd);

    return {remaining, outputMemBlocks};
}

static void AllocateBaseEdges(const EdgeClusters& edgeClusters, const MemoryControl::MemorySolution& memorySolution) {
    // attach all the not yet allocated edges to the memory control
    for (const auto& [regionId, memoryBlock] : memorySolution) {
        const auto& cluster = edgeClusters.at(regionId);
        const auto& baseEdge = cluster.at(0);

        baseEdge->allocate(memoryBlock);
        // TODO: WA for some test (like strided_slice_test) which use tensors with
        //       shapes {0}. And it is implicitly converted into {1} tensor.
        //       Zeroing of input data allow pass tests.
        if (baseEdge->getParent()->getType() == Type::Input && baseEdge->getMemory().getDesc().hasDefinedMaxSize()) {
            baseEdge->getMemoryPtr()->nullify();
        }
    }
}

static void AllocatedReferencingEdges(const EdgeClusters& clusters) {
    for (const auto& cluster : clusters) {
        // reuse starting from second edge (skip the base edge)
        for (size_t i = 1; i < cluster.size(); i++) {
            const auto& edge = cluster.at(i);
            // child edges on the same port are in the same cluster
            // and allocated in scope of single resolveInPlaceEdges call
            if (edge->getStatus() == Edge::Status::Allocated) {
                continue;
            }

            if (edge->inPlace(Edge::LOOK_DOWN)) {
                edge->getChild()->resolveInPlaceEdges(Edge::LOOK_DOWN);
            } else if (edge->inPlace(Edge::LOOK_UP)) {
                edge->getParent()->resolveInPlaceEdges(Edge::LOOK_UP);
            } else {
                auto sharedEdge = edge->getSharedEdge();
                edge->allocate(sharedEdge->getMemoryPtr()->getMemoryBlock());
                DEBUG_LOG(*edge, " sharedEdge with ", *sharedEdge);
            }
        }
    }
}

std::vector<size_t> Graph::CreateExecutionGraph() {
    const bool hasDynNodes = ProcessDynNodes();
    auto syncNodesInds = hasDynNodes ? IdentifySyncPoints(graphNodes) : std::vector<size_t>{};

    std::tie(m_executableGraphNodes, m_executableSyncNodesInds) =
        ExtractExecutableNodesAndSyncPoints(syncNodesInds, graphNodes);

    if (hasDynNodes) {
        status = Status::ReadyDynamic;
        // Here we use the following heuristic: if the number of sync nodes is less than 10 times of the number of exec
        // nodes, it does make sense to use Sequential dynamic shapes processing due to the high overheads on context
        // switching when the dynamic shapes are being processed in parallel and there are a lot of sync points. Also
        // this rule works for short graphs (usually subgraphs) when the amount of nodes is to low to process them in
        // parallel.
        const auto exec2sync = m_executableGraphNodes.size() / m_executableSyncNodesInds.size();
        if (exec2sync < 10 || parallel_get_max_threads() < 2) {
            status = Status::ReadyDynamicSeq;
        }
    } else {
        status = Status::ReadyStatic;
    }

    return syncNodesInds;
}

static void ResolveInOutInPlaceEdges(const std::vector<EdgePtr>& edges) {
    for (const auto& edge : edges) {
        if (edge->getStatus() == Edge::Status::Uninitialized) {
            if (edge->getParent()->getParentEdges().empty() &&
                any_of(edge->getParent()->getType(), Type::MemoryInput) && edge->inPlace(Edge::LOOK_UP)) {
                edge->getParent()->resolveInPlaceEdges(Edge::LOOK_UP);
            } else if (edge->getChild()->getChildEdges().empty() &&
                       any_of(edge->getChild()->getType(), Type::MemoryOutput) && edge->inPlace(Edge::LOOK_DOWN)) {
                edge->getChild()->resolveInPlaceEdges(Edge::LOOK_DOWN);
            }
        }
    }
}

int Graph::RegisterToAllocationContext(int offset, AllocationContext& context) {
    auto syncNodesInds = CreateExecutionGraph();

    ResolveInOutInPlaceEdges(graphEdges);

    // nodes are expected to be topologically sorted
    for (size_t execIndex = 0, syncNodeIdx = 0; execIndex < graphNodes.size(); execIndex++) {
        const auto& node = graphNodes[execIndex];
        const auto inputExecIndex = offset;
        // register local sync node idx to global allocation context as well
        if (syncNodeIdx < syncNodesInds.size() && syncNodesInds[syncNodeIdx] == execIndex) {
            context.syncPoints.push_back(inputExecIndex);
            syncNodeIdx++;
        }

        // an offset is the number of nodes in the internal graph minus the current node (-1)
        offset = node->registerToAllocationContext(inputExecIndex, context);
        const auto outputExecIndex = offset;
        offset++;
        context.execIndex[node] = {inputExecIndex, outputExecIndex};
    }

    context.edges.insert(context.edges.end(), graphEdges.begin(), graphEdges.end());

    return offset - 1;
}

static void InitEdgeStatus(const std::vector<EdgePtr>& edges) {
    for (const auto& edge : edges) {
        edge->init();
    }
}

static void ValidateEdgeStatus(const std::vector<EdgePtr>& edges) {
    for (const auto& edge : edges) {
        edge->validate();
    }
}

/**
 * Forms clusters of edges.
 * An edge cluster is a collection of edges, with the following properties:
 * - base edge is an edge with a Memory which other edges point to by means of inplace logic
 * - first edge of a cluster is a base edge with a status NeedAllocation
 * - rest of the edges in a cluster are NotAllocated ones and they point to the base edge
 *
 * A cluster essentially looks like:
 * - std::vector<EdgePtr> cluster{base, edge1, edge2, edge3, ...}
 */
static EdgeClusters FormEdgeClusters(const std::vector<EdgePtr>& graphEdges) {
    using EdgeClusterIdxMap = std::unordered_map<EdgePtr, size_t>;
    EdgeClusters edgeClusters;
    EdgeClusterIdxMap edgeClusterIndices;

    std::function<size_t(EdgePtr)> addToCluster;
    addToCluster = [&addToCluster, &edgeClusterIndices, &edgeClusters](const EdgePtr& edge) {
        // cluster is already created for the edge
        if (auto it = edgeClusterIndices.find(edge); it != edgeClusterIndices.end()) {
            return it->second;
        }
        // create an edge cluster when the base edge is visited for the first time
        // so the base edge is always the first edge in the cluster
        if (edge == nullptr) {
            edgeClusters.emplace_back();
            return edgeClusters.size() - 1;
        }

        auto clusterIdx = addToCluster(edge->getSharedEdge(std::nothrow));

        edgeClusterIndices[edge] = clusterIdx;
        edgeClusters[clusterIdx].push_back(edge);

        return clusterIdx;
    };

    for (const auto& edge : graphEdges) {
        [[maybe_unused]] const auto clusterIdx = addToCluster(edge);
        DEBUG_LOG("Added edge: ", *edge, " to cluster: ", clusterIdx);
    }

    return edgeClusters;
}

/**
 * @brief Validates the correctness of edge clusters.
 *
 * This function ensures that each edge cluster follows the expected structure:
 * - The cluster must not be empty.
 * - The first edge in the cluster (base edge) must have the status `NeedAllocation`.
 * - All subsequent edges in the cluster must have the status `NotAllocated`.
 *
 * @param clusters   collection of edges.
 * @param remaining  number of clusters to process (not allocated yet).
 *
 * @throws If any of the required conditions are violated
 */
static void ValidateEdgeClusters(const EdgeClusters& clusters, size_t remaining) {
    for (size_t i = 0; i < remaining; i++) {
        const auto& cluster = clusters.at(i);
        OPENVINO_ASSERT(!cluster.empty(), "Unexpected empty edge cluster");

        const auto& baseEdge = cluster.at(0);
        OPENVINO_ASSERT(baseEdge->getStatus() == Edge::Status::NeedAllocation,
                        "Unexpected status of edge: ",
                        *baseEdge);

        for (size_t i = 1; i < cluster.size(); ++i) {
            const auto& edge = cluster.at(i);
            OPENVINO_ASSERT(edge->getStatus() == Edge::Status::NotAllocated, "Unexpected status of edge: ", *edge);
        }
    }
}

static MemoryRegions FormMemoryRegions(const EdgeClusters& clusters,
                                       size_t remaining,
                                       const GlobalExecutionIndex& globalExecIndex) {
    auto isConstOutput = [](const EdgePtr& edge) {
        return edge->getParent()->isConstant() && !edge->getChild()->isConstant();
    };

    // Markup the memory regions
    MemoryRegions memoryRegions;
    memoryRegions.reserve(remaining);

    for (size_t i = 0; i < remaining; ++i) {
        MemoryRegion reg = {std::numeric_limits<int>::max(),
                            0,
                            0,
                            static_cast<int64_t>(i),
                            MemoryRegion::RegionType::VARIABLE,
                            MemoryRegion::AllocType::UNKNOWN};

        int64_t boxSize = 0;
        bool isConst = false;
        bool isOutput = false;
        bool isInput = false;

        for (const auto& edge : clusters[i]) {
            const auto& parent = edge->getParent();
            const auto& child = edge->getChild();

            auto usesInOutMemoryMultipleTimes = [](const NodePtr& node) {
                if (auto tensorIterator = std::dynamic_pointer_cast<node::TensorIterator>(node)) {
                    return tensorIterator->usesInOutMemoryMultipleTimes();
                }

                return false;
            };
            // If node uses its input / output memory multiple times in scope of a single execution (i.e TensorIterator)
            // prolong the lifetime of a memory region till execution is finished
            int e_start = usesInOutMemoryMultipleTimes(parent) ? globalExecIndex.at(parent).first
                                                               : globalExecIndex.at(parent).second;
            int e_finish = usesInOutMemoryMultipleTimes(child) ? globalExecIndex.at(child).second
                                                               : globalExecIndex.at(child).first;

            auto&& desc = edge->getOriginalDesc();

            if (boxSize != -1 && desc.isDefined()) {
                int64_t e_size =
                    desc.getCurrentMemSize();  // size in bytes (from the beginning of data to the last element)
                boxSize = std::max(e_size, boxSize);
            } else {
                boxSize = -1;
            }

            reg.start = std::min(e_start, reg.start);
            reg.finish = std::max(e_finish, reg.finish);

            auto allocType =
                desc.getPrecision() == element::string ? MemoryRegion::AllocType::STRING : MemoryRegion::AllocType::POD;

            OPENVINO_ASSERT(any_of(reg.alloc_type, allocType, MemoryRegion::AllocType::UNKNOWN),
                            "Different allocation types in the same memory region");
            reg.alloc_type = allocType;

            isConst |= isConstOutput(edge);
            isOutput |= child->getType() == Type::Output;
            isInput |= parent->getType() == Type::Input;
        }

        reg.size = boxSize;

        if (isConst) {
            reg.type = MemoryRegion::RegionType::CONSTANT;
        } else if (isInput) {
            if (isOutput) {
                reg.type = MemoryRegion::RegionType::IO;
            } else {
                reg.type = MemoryRegion::RegionType::INPUT;
            }
        } else if (isOutput) {
            reg.type = MemoryRegion::RegionType::OUTPUT;
        }

        memoryRegions.push_back(reg);
    }

    return memoryRegions;
}

static size_t SkipAllocatedClusters(EdgeClusters& clusters) {
    auto notAllocatedPartitionEnd = std::partition(clusters.begin(), clusters.end(), [](const EdgeCluster& cluster) {
        const auto& baseEdge = cluster.at(0);

        return baseEdge->getStatus() != Edge::Status::Allocated;
    });

    return std::distance(clusters.begin(), notAllocatedPartitionEnd);
}

/**
 * Solve memory reuse
 * Ideally only MemorySolution should be returned
 * For now we have to additionally return:
 * 1) EdgeClusters - to propagate the solution through the graph
 * 2) OutputMemoryBlocks - to allow memory sharing between graph and infer request
 */
static std::tuple<MemoryControl::MemorySolution, EdgeClusters, Graph::OutputMemoryBlocks> SolveMemoryReuse(
    const std::shared_ptr<MemoryControl>& memoryControl,
    const AllocationContext& allocationContext,
    const GraphContext::CPtr& graphContext,
    const std::vector<NodePtr>& outputNodes) {
    const auto& edges = allocationContext.edges;

    auto edgeClusters = FormEdgeClusters(edges);

    size_t remaining = SkipAllocatedClusters(edgeClusters);

    ValidateEdgeClusters(edgeClusters, remaining);
    // strings and constant are allocated bypassing the memory control
    // @todo allocate constants using dedicated memory control
    remaining = AllocateStringsAndConstants(edgeClusters, remaining, graphContext);
    // dynamic output edges are allocated bypassing the memory control
    Graph::OutputMemoryBlocks outputNodesMemBlocks;
    std::tie(remaining, outputNodesMemBlocks) = AllocateDynamicOutputEdges(edgeClusters, remaining, outputNodes);

    auto memoryRegions = FormMemoryRegions(edgeClusters, remaining, allocationContext.execIndex);

    memoryControl->insert(memoryRegions, allocationContext.syncPoints);
    auto memoryBlocks = memoryControl->solve();

    return std::make_tuple(memoryBlocks, edgeClusters, outputNodesMemBlocks);
}

void Graph::Allocate() {
    auto memoryControl = m_context->getMemoryControl();

    if (memoryControl->allocated()) {
        return;  // memory is already allocated globally
    }

    AllocationContext allocationContext;
    RegisterToAllocationContext(0, allocationContext);

    const auto& edges = allocationContext.edges;
    InitEdgeStatus(edges);

    MemoryControl::MemorySolution solution;
    EdgeClusters edgeClusters;
    std::tie(solution, edgeClusters, m_outputNodesMemBlocks) =
        SolveMemoryReuse(memoryControl, allocationContext, m_context, outputNodes);

    AllocateBaseEdges(edgeClusters, solution);

    memoryControl->allocateMemory();

    AllocatedReferencingEdges(edgeClusters);

    ValidateEdgeStatus(edges);
}

bool Graph::ProcessDynNodes() const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::ov_intel_cpu_LT, "Graph::ProcessDynNodes");

    const bool containsDynamicNodes = std::any_of(graphNodes.begin(), graphNodes.end(), [](const NodePtr& node) {
        return node->isDynamicNode();
    });

    return containsDynamicNodes;
}

void Graph::PushInputData(const std::size_t& index, const ov::SoPtr<ITensor>& input) {
    OPENVINO_ASSERT(IsReady(), "Wrong state. Topology not ready.");
    if (index < inputNodes.size() && inputNodes[index]) {
        auto node = inputNodes[index];
        auto childEdge = node->getChildEdgeAt(0);
        const auto& edgeMemory = childEdge->getMemory();

        const void* ext_data_ptr = input->data();
        void* inter_data_ptr = edgeMemory.getData();

        if (ext_data_ptr != inter_data_ptr) {
            auto ext_tensor_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(input);
            auto actualDesc = edgeMemory.getDescPtr();

            if (actualDesc->getPrecision() == element::string) {
                StringMemory ext_mem(getEngine(), ext_tensor_desc, ext_data_ptr);
                edgeMemory.load(ext_mem, false, false);
            } else if (!actualDesc->isCompatible(*ext_tensor_desc)) {
                Memory ext_mem(getEngine(), ext_tensor_desc, ext_data_ptr, false);
                edgeMemory.load(ext_mem, false, false);
            } else {
                size_t size_to_copy = ext_tensor_desc->getCurrentMemSize();
                cpu_parallel_memcpy(inter_data_ptr, ext_data_ptr, size_to_copy);
            }
        }
    } else {
        OPENVINO_THROW("Input tensor with index '", index, "' is not available in the model");
    }
}

// suppose always being shared infer_request intel_cpu::Tensor to Graph if isDynamic.
void Graph::PullOutputData(std::unordered_map<std::size_t, ov::SoPtr<ITensor>>& output) {
    OPENVINO_ASSERT(IsReady(), "Wrong state. Topology not ready.");

    for (size_t output_index = 0; output_index < outputNodes.size(); ++output_index) {
        auto node = outputNodes[output_index];
        auto parentEdge = node->getParentEdgeAt(0);
        const auto& intr_blob = parentEdge->getMemory();

        auto output_itr = output.find(output_index);
        OPENVINO_ASSERT(output_itr != output.end(),
                        "The CPU plugin graph doesn't contain output node with output_index: ",
                        output_index);
        const auto ext_blob = output_itr->second;

        DEBUG_LOG(output_index, ", tensor data addr ", static_cast<void*>(output[output_index]->data()));

        // TODO [NM]: need to create a universal reorder which will detect cases when we really need to use it
        // WA: for cases when output shape after transformation is 1x1x1x1 but the model output is scalar
        const auto& actualDims = ext_blob->get_shape();
        const auto& outDims = intr_blob.getStaticDims();

        const bool isScalarOutput = actualDims.empty() && 1 == ext_blob->get_size();

        if (!isScalarOutput && actualDims != outDims) {
            // WA: because input/output info initially contains non empty dims, order etc.
            // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
            DEBUG_LOG(output_index,
                      ", tensor data addr ",
                      static_cast<void*>(output[output_index]->data()),
                      " dims ",
                      PartialShape(output[output_index]->get_shape()),
                      " -> ",
                      PartialShape(outDims),
                      ", intr ptr ",
                      intr_blob.getData(),
                      " , parentedge's memory object ",
                      parentEdge->getMemoryPtr().get());
            ext_blob->set_shape(outDims);
            DEBUG_LOG(output_index,
                      ", tensor data addr ",
                      static_cast<void*>(output[output_index]->data()),
                      " dims ",
                      PartialShape(output[output_index]->get_shape()),
                      ", intr ptr ",
                      intr_blob.getData());
        }

        // check for empty output blob
        if (std::any_of(outDims.begin(), outDims.end(), [](const Dim dim) {
                return dim == 0;
            })) {
            continue;
        }

        auto srcPrec = intr_blob.getPrecision();
        auto dstPrec = ext_blob->get_element_type();
        OPENVINO_ASSERT(srcPrec != dstPrec || ext_blob->get_byte_size() == intr_blob.getSize(),
                        "Output tensor byte size is not equal model output byte size (",
                        ext_blob->get_byte_size(),
                        "!=",
                        intr_blob.getSize(),
                        ").");

        void* ext_blob_ptr = ext_blob->data();
        void* intr_blob_ptr = intr_blob.getData();
        DEBUG_LOG(output_index,
                  " @ ",
                  intr_blob_ptr,
                  " -> ",
                  ext_blob_ptr,
                  " zero-copy: ",
                  intr_blob_ptr == ext_blob_ptr,
                  " graph ",
                  this,
                  "\r\n");

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) {
            continue;
        }

        auto externDesc = MemoryDescUtils::generateCpuBlockedMemoryDesc(ext_blob);
        auto actualDesc = intr_blob.getDescWithType<BlockedMemoryDesc>();
        if (actualDesc->getPrecision() == element::string) {
            StringMemory outBloMem(getEngine(), externDesc, ext_blob_ptr);
            outBloMem.load(intr_blob, false, false);
        } else if (!actualDesc->isCompatible(*externDesc) && !isScalarOutput) {
            Memory outBloMem(getEngine(), externDesc, ext_blob_ptr, false);
            outBloMem.load(intr_blob, false, false);
        } else {
            OPENVINO_ASSERT(srcPrec == dstPrec,
                            "The precision of the CPU output tensor index",
                            output_index,
                            " is different from the external one");
            size_t size_to_copy = intr_blob.getSize();
            cpu_parallel_memcpy(ext_blob_ptr, intr_blob_ptr, size_to_copy);
        }
    }
}

VecMemoryDescs Graph::getOutputMemoryDescriptors() const {
    OPENVINO_ASSERT(status == Status::Initialized, "Invalid graph status");

    VecMemoryDescs result;
    result.reserve(outputNodes.size());

    for (const auto& node : outputNodes) {
        result.emplace_back(node->getBaseMemDescAtInputPort(0));
    }

    return result;
}

void Graph::InferStatic(SyncInferRequest* request, int numaId) {
    for (const auto& node : m_executableGraphNodes) {
        ExecuteNodeWithCatch(node, request, numaId);
    }
}

namespace {

class UpdateNodesSeq {
public:
    explicit UpdateNodesSeq(std::vector<NodePtr>& executableGraphNodes)
        : m_executableGraphNodes(executableGraphNodes) {}

    void operator()(size_t stopIndx) {
        for (; prepareCounter < stopIndx; ++prepareCounter) {
            const auto& node = m_executableGraphNodes[prepareCounter];
            if (node->isDynamicNode()) {
                node->updateShapes();
                node->updateDynamicParams();
            }
        }
    }

private:
    size_t prepareCounter = 0;
    std::vector<NodePtr>& m_executableGraphNodes;
};

#if (OV_THREAD == OV_THREAD_SEQ)
using UpdateNodes = UpdateNodesSeq;
#endif

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE || \
     OV_THREAD == OV_THREAD_OMP)

class UpdateNodesBase {
public:
    explicit UpdateNodesBase(std::vector<NodePtr>& executableGraphNodes)
        : m_executableGraphNodes(executableGraphNodes) {}
    void updateShapes(size_t node_indx, size_t stop_indx) {
        try {
            for (size_t i = node_indx; i < stop_indx; i++) {
                const auto& node = m_executableGraphNodes[i];
                if (node->isDynamicNode()) {
                    node->updateShapes();
                }
                m_prepareCounter.store(i, std::memory_order_release);
            }
        } catch (...) {
            m_completion.store(true, std::memory_order_release);
            throw;
        }
        m_prepareCounter.store(stop_indx, std::memory_order_relaxed);
        m_completion.store(true, std::memory_order_release);
    }

    void updateDynParams(size_t node_indx, [[maybe_unused]] size_t stop_indx) {
        size_t local_counter = node_indx;
        while (true) {
            const bool completion = m_completion.load(std::memory_order_acquire);
            const size_t prepareCounter = m_prepareCounter.load(std::memory_order_relaxed);
            if (completion && local_counter == prepareCounter) {
                break;
            }
            while (local_counter < prepareCounter) {
                const auto& node = m_executableGraphNodes[local_counter++];
                if (node->isDynamicNode()) {
                    node->updateDynamicParams();
                }
            }
        }
    }

protected:
    std::atomic<size_t> m_prepareCounter{0};
    std::atomic<bool> m_completion{false};
    std::vector<NodePtr>& m_executableGraphNodes;
};

// NOLINTBEGIN(misc-include-cleaner) tbb has multiple implicit includes, which are not supposed to be included directly
#    if OV_THREAD_USE_TBB
#        if (TBB_VERSION_MAJOR > 2020)
template <typename Body>
class AsyncTask : public tbb::detail::d1::task {
public:
    AsyncTask(Body& body, tbb::detail::d1::wait_context& wait, size_t node_indx, size_t stop_indx)
        : m_body(body),
          m_wait(wait),
          m_node_indx(node_indx),
          m_stop_indx(stop_indx) {}
    task* execute([[maybe_unused]] tbb::detail::d1::execution_data& data) override {
        m_body(m_node_indx, m_stop_indx);
        m_wait.release();
        return nullptr;