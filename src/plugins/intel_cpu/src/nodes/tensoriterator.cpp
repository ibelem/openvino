// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensoriterator.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "allocation_context.hpp"
#include "cache/multi_cache.h"
#include "common/blocked_desc_creator.h"
#include "common/cpu_memcpy.h"
#include "common/reorder_prim.h"
#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

static NodeConfig make_plain_config(const std::shared_ptr<ov::Node>& op) {
    NodeConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto& origShape = op->get_input_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = op->get_input_element_type(i);

        PortConfig data_conf{};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto& origShape = op->get_output_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = op->get_output_element_type(i);

        PortConfig data_conf{};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.outConfs.push_back(data_conf);
    }

    return config;
}

static void redefineToMemories(const std::vector<MemoryPtr>& to_mems, const MemoryDescPtr& new_desc) {
    // TODO : check the entire dstMemPtrs usage considering the proper memory sharing
    for (const auto& to_mem : to_mems) {
        to_mem->redefineDesc(new_desc);
    }
}

// this method get all memory ptrs of childs of one port to redefine descs for them
static std::vector<MemoryPtr> getToMemories(const Node* node, const size_t port) {
    std::vector<MemoryPtr> memories;
    for (auto& edge : node->getChildEdgesAtPort(port)) {
        memories.push_back(edge->getMemoryPtr());
    }
    return memories;
}

static void nullifyUndefinedDims(VectorDims& dims) {
    std::transform(dims.begin(), dims.end(), dims.begin(), [](const size_t& dim) {
        return dim == Shape::UNDEFINED_DIM ? 0 : dim;
    });
}

static bool hasEmptyDims(const dnnl::memory& mem) {
    const auto& dims = mem.get_desc().get_dims();
    return std::any_of(dims.begin(), dims.end(), [](const dnnl::memory::dim dim) {
        return dim == 0;
    });
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(const MultiCachePtr& cache,
                       const MemoryPtr& from,
                       const MemoryPtr& to,
                       bool sliced_src,
                       const PortMap& slice_rule,
                       const dnnl::engine& eng)
        : sliced_src(sliced_src) {
        const auto& full_blob = sliced_src ? from : to;
        const auto& part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.axis;
        auto stride = slice_rule.stride;

        auto full_dims = full_blob->getShape().getStaticDims();
        auto part_dims = part_blob->getShape().getStaticDims();

        auto abs_stride = std::abs(stride);
        OPENVINO_ASSERT(abs_stride != 0, "TensorIterator PortMap stride must not be zero");
        auto sign_of_stride = stride < 0 ? -1 : 1;

        iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        OPENVINO_ASSERT(full_dims == part_dims, "Shape mismatch for tensor iterator port");

        // make chunk view
        auto chunk_desc = full_blob->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        chunk_desc.get()->dims[axis] = abs_stride;
        chunk_desc.get()->padded_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        full_mem = full_blob->getPrimitive();
        auto* const full_mem_handler = full_mem.get_data_handle();
        dnnl::memory chunk_mem = {chunk_desc, eng, full_mem_handler};

        auto elem_size = DnnlExtensionUtils::sizeOfDataType(chunk_desc.get_data_type());

        chunk_stride_in_byte = chunk_desc.get()->format_desc.blocking.strides[axis] * elem_size * abs_stride;
        chunk_offset_in_byte = sign_of_stride < 0 ? (iter_count - 1) * chunk_stride_in_byte : 0;
        chunk_stride_in_byte *= sign_of_stride;

        if (sliced_src) {
            mem_holder_src = chunk_mem;
            mem_holder_dst = to->getPrimitive();
        } else {
            mem_holder_src = from->getPrimitive();
            mem_holder_dst = chunk_mem;
        }
        reorder =
            getReorderPrim(cache, mem_holder_dst.get_engine(), mem_holder_src.get_desc(), mem_holder_dst.get_desc());
    }

    void execute(const dnnl::stream& strm, int iter) override {
        OPENVINO_ASSERT(iter >= 0 && iter < iter_count);

        if (hasEmptyDims(mem_holder_src) || hasEmptyDims(mem_holder_dst)) {
            return;
        }

        auto& chunk_mem = sliced_src ? mem_holder_src : mem_holder_dst;
        chunk_mem.set_data_handle(static_cast<uint8_t*>(full_mem.get_data_handle()) + chunk_offset_in_byte +
                                  chunk_stride_in_byte * iter);

        reorder.execute(strm, {{DNNL_ARG_FROM, mem_holder_src}, {DNNL_ARG_TO, mem_holder_dst}});
    }

private:
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    bool sliced_src;
    dnnl::memory full_mem;

    int iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MultiCachePtr& cache, const MemoryPtr& from, const MemoryPtr& to) {
        mem_holder_src = from->getPrimitive();
        mem_holder_dst = to->getPrimitive();
        reorder =
            getReorderPrim(cache, mem_holder_dst.get_engine(), mem_holder_src.get_desc(), mem_holder_dst.get_desc());
    }

    void execute(const dnnl::stream& strm, int iter) override {
        if (iter != 0) {
            if (hasEmptyDims(mem_holder_src) || hasEmptyDims(mem_holder_dst)) {
                return;
            }

            reorder.execute(strm, {{DNNL_ARG_FROM, mem_holder_src}, {DNNL_ARG_TO, mem_holder_dst}});
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MemoryPtr& to, [[maybe_unused]] const dnnl::engine& eng) {
        // Only scalar I32 tensor is supported
        OPENVINO_ASSERT(to->getDataType() == memory::data_type::s32);
        OPENVINO_ASSERT(to->getShape() == Shape(VectorDims{1}));
        mem_holder_dst = to->getPrimitive();
    }

    void execute([[maybe_unused]] const dnnl::stream& strm, int n_iter) override {
        auto mem = mem_holder_dst;
        auto* data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        OPENVINO_ASSERT(data_ptr, "TensorIterator node has not allocated memory for IterCountPortHelper");
        *data_ptr = n_iter;
    }
};

class asBoolCheck : public PortChecker {
public:
    explicit asBoolCheck(const MemoryPtr& mem) {
        OPENVINO_ASSERT(mem->getDataType() == memory::data_type::u8);
        OPENVINO_ASSERT(mem->getShape() == Shape(VectorDims{1}));
        mem_holder = mem->getPrimitive();
    }

    int getStatus() override {
        auto* data_ptr = static_cast<uint8_t*>(mem_holder.get_data_handle());
        OPENVINO_ASSERT(data_ptr, "TensorIterator node has not allocated memory for asBoolCheck");
        return *data_ptr == static_cast<uint8_t>(0) ? 0 : 1;
    }
};

class asIntCheck : public PortChecker {
public:
    explicit asIntCheck(const MemoryPtr& mem) {
        OPENVINO_ASSERT(mem->getDataType() == memory::data_type::s32);
        OPENVINO_ASSERT(mem->getShape() == Shape(VectorDims{1}));
        mem_holder = mem->getPrimitive();
    }

    int getStatus() override {
        auto* data_ptr = static_cast<uint32_t*>(mem_holder.get_data_handle());
        OPENVINO_ASSERT(data_ptr, "TensorIterator node has not allocated memory for asIntCheck");
        return *data_ptr;
    }
};

class staticValueCheck : public PortChecker {
public:
    explicit staticValueCheck(const int& value) : value(value) {}

    int getStatus() override {
        return value;
    }

private:
    int value;
};

DynamicBuffer::DynamicBuffer(MemoryPtr from_,
                             std::vector<MemoryPtr> to_,
                             const PortMap& map_rule_,
                             const std::shared_ptr<CpuParallel>& parallel)
    : from(std::move(from_)),
      to(std::move(to_)),
      map_rule(map_rule_),
      elem_size(DnnlExtensionUtils::sizeOfDataType(from->getDataType())),
      cpu_parallel(parallel) {}

void DynamicBuffer::execute(const dnnl::engine& eng, const int iter) {
    OPENVINO_ASSERT(from->getStaticDims()[map_rule.axis] == static_cast<size_t>(std::abs(map_rule.stride)),
                    "TensorIterator (Loop) has incorrect output shape[axis] after iteration for concatenation. ",
                    std::abs(map_rule.stride),
                    " is expected, but actual: ",
                    from->getStaticDims()[map_rule.axis]);

    if (iter == 0) {
        init(eng);
    }

    // if chunk_offset_in_byte out of range of buffer holder, reallocate a larger chunk
    if (check_buffer()) {
        auto new_buffer = create_buffer(eng);
        move_buffer(new_buffer);
    }

    move_data();
}

void DynamicBuffer::reset(int max_iter_count_) {
    max_iter_count = max_iter_count_;
}

void DynamicBuffer::init(const dnnl::engine& eng) {
    const auto stride = map_rule.stride;
    const auto abs_stride = std::abs(stride);

    // We have no idea of "from" node memory dims until the sub_graph has been executed.
    const auto& src_mem = from->getPrimitive();
    const auto& src_desc = src_mem.get_desc();
    const auto& dims = src_desc.get_dims();
    count = std::accumulate(dims.begin(), dims.begin() + map_rule.axis, static_cast<size_t>(1), std::multiplies<>());
    len = std::accumulate(dims.begin() + map_rule.axis + 1, dims.end(), elem_size, std::multiplies<>());
    chunk_unit_in_byte = abs_stride * len;

    if (!mem_holder_buffer) {  // else reuse buffer holder of last inference
        // preallocate a large chunk of memory to hold intermediate concated outputs of all iterations.
        mem_holder_buffer = create_buffer(eng);
    }

    // reset chunk_offset_in_byte since the first execution
    chunk_stride_in_byte = mem_holder_buffer->getSize() / count;
    chunk_offset_in_byte = stride > 0 ? 0 : (chunk_stride_in_byte - chunk_unit_in_byte);
    num_execs = 0;
}

bool DynamicBuffer::check_buffer() const {
    if (map_rule.stride > 0) {
        if (static_cast<ptrdiff_t>(chunk_offset_in_byte + chunk_unit_in_byte) > chunk_stride_in_byte) {
            return true;
        }
    } else {
        if (chunk_offset_in_byte < 0) {
            return true;
        }
    }
    return false;
}

MemoryPtr DynamicBuffer::create_buffer(const dnnl::engine& eng) {
    const auto abs_stride = std::abs(map_rule.stride);

    const auto estimate_iters = [&]() {
        if (max_iter_count != -1) {
            return max_iter_count;
        }

        // in case of no idea of memory upper boundary
        return (num_execs == 0) ? 1 : 2 * num_execs;  // growth factor 2
    };
    const auto estimated_iters = estimate_iters();
    const Shape _shape = Shape({count, static_cast<size_t>(abs_stride * estimated_iters), len / elem_size});
    auto _descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
    auto new_buffer_desc = _descCreator->createSharedDesc(from->getDesc().getPrecision(), _shape);

    auto _ptr = std::make_shared<Memory>(eng, new_buffer_desc);
    return _ptr;
}

void DynamicBuffer::move_buffer(const MemoryPtr& new_buffer) {
    const auto stride = map_rule.stride;

    // copy data from old buffer to new buffer
    const auto src_stride = chunk_stride_in_byte;
    const auto dst_stride = new_buffer->getStaticDims()[1] * len;

    const auto valid_size = chunk_unit_in_byte * num_execs;
    const auto src_offset_in_byte = stride > 0 ? 0 : (src_stride - valid_size);
    chunk_offset_in_byte = stride > 0 ? 0 : (dst_stride - valid_size);  // reset chunk_offset_in_byte

    copy(mem_holder_buffer->getDataAs<uint8_t>() + src_offset_in_byte,
         new_buffer->getDataAs<uint8_t>() + chunk_offset_in_byte,
         src_stride,
         dst_stride,
         count,
         valid_size,
         cpu_parallel);

    // assign mem_holder_buffer
    mem_holder_buffer = new_buffer;
    chunk_stride_in_byte = mem_holder_buffer->getSize() / count;

    // adjust for next execution
    if (stride > 0) {
        chunk_offset_in_byte += valid_size;
    } else {
        chunk_offset_in_byte -= chunk_unit_in_byte;
    }
}

void DynamicBuffer::move_data() {
    const auto src_stride = abs(map_rule.stride) * len;
    const auto dst_stride = chunk_stride_in_byte;

    copy(from->getDataAs<const uint8_t>(),
         mem_holder_buffer->getDataAs<uint8_t>() + chunk_offset_in_byte,
         src_stride,
         dst_stride,
         count,
         chunk_unit_in_byte,
         cpu_parallel);

    // adjust for next execution
    num_execs++;
    if (map_rule.stride > 0) {
        chunk_offset_in_byte += chunk_unit_in_byte;
    } else {
        chunk_offset_in_byte -= chunk_unit_in_byte;
    }
}

void DynamicBuffer::transfer(const Node* node) {
    if (mem_holder_buffer && num_execs > 0) {
        const auto axis = map_rule.axis;
        const auto stride = map_rule.stride;
        const auto abs_stride = std::abs(stride);

        const auto& src_mem = from->getPrimitive();
        const auto& src_desc = src_mem.get_desc();
        auto dims = src_desc.get_dims();
        dims[axis] = abs_stride * num_execs;
        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)
                              ->cloneWithNewDims(DnnlExtensionUtils::convertToVectorDims(dims));

        redefineToMemories(to, desc);

        const auto src_stride = chunk_stride_in_byte;
        const auto dst_stride = to.front()->getStaticDims()[axis] * len;
        const auto valid_size = chunk_unit_in_byte * num_execs;
        const auto src_offset_in_byte = stride > 0 ? 0 : (src_stride - valid_size);

        copy(mem_holder_buffer->getDataAs<uint8_t>() + src_offset_in_byte,
             to.front()->getDataAs<uint8_t>(),
             src_stride,
             dst_stride,
             count,
             dst_stride,
             cpu_parallel);
    } else {
        VectorDims newDims = to.front()->getShape().getDims();
        nullifyUndefinedDims(newDims);

        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(newDims);
        redefineToMemories(to, desc);
    }
}

void DynamicBuffer::copy(const uint8_t* src,
                         uint8_t* dst,
                         const size_t src_stride,
                         const size_t dst_stride,