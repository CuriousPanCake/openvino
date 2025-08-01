// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_convert.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NECast.h>
#include <arm_compute/runtime/NEON/functions/NECopy.h>

#include <cassert>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/convert.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using namespace arm_compute;

bool ACLConvertExecutor::init(const ConvertParams& convertParams,
                              [[maybe_unused]] const MemoryDescPtr& srcDesc,
                              [[maybe_unused]] const MemoryDescPtr& dstDesc,
                              [[maybe_unused]] const dnnl::primitive_attr& attr) {
    aclConvertParams = convertParams;

    auto srcPrecision = precisionToAclDataType(aclConvertParams.srcPrc);
    auto dstPrecision = precisionToAclDataType(aclConvertParams.dstPrc);
    isCopyOp = aclConvertParams.srcPrc == aclConvertParams.dstPrc;
    // NECast does not support S8. It could be replaced with QASYMM8_SIGNED
    if (!isCopyOp && srcPrecision == DataType::S8) {
        srcPrecision = DataType::QASYMM8_SIGNED;
    }
    if (!isCopyOp && dstPrecision == DataType::S8) {
        dstPrecision = DataType::QASYMM8_SIGNED;
    }
    // Use 1D TensorInfo, since UNKNOWN DataLayout may have accuracy issues
    auto srcDims1D = convertParams.size;
    auto dstDims1D = convertParams.size;
    auto srcTensorInfo = TensorInfo(TensorShape(srcDims1D), 1, srcPrecision);
    auto dstTensorInfo = TensorInfo(TensorShape(dstDims1D), 1, dstPrecision);
    if (isCopyOp) {
        Status s = NECopy::validate(&srcTensorInfo, &dstTensorInfo);
        if (!s) {
            DEBUG_LOG("NECopy validation failed: ", s.error_description());
            return false;
        }
    } else {
        Status s = NECast::validate(&srcTensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE);
        if (!s) {
            DEBUG_LOG("NECast validation failed: ", s.error_description());
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (isCopyOp) {
        acl_copy = std::make_unique<NECopy>();
        configureThreadSafe([&] {
            acl_copy->configure(&srcTensor, &dstTensor);
        });
    } else {
        acl_cast = std::make_unique<NECast>();
        configureThreadSafe([&] {
            acl_cast->configure(&srcTensor, &dstTensor, ConvertPolicy::SATURATE);
        });
    }
    return true;
}

void ACLConvertExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    assert(src.size() == 1);
    assert(dst.size() == 1);

    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    if (isCopyOp) {
        acl_copy->run();
    } else {
        acl_cast->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool ACLConvertExecutorBuilder::isSupported(const ConvertParams& convertParams,
                                            [[maybe_unused]] const MemoryDescPtr& srcDesc,
                                            [[maybe_unused]] const MemoryDescPtr& dstDesc) const {
    if (convertParams.srcPrc != convertParams.dstPrc) {
        if (none_of(convertParams.srcPrc,
                    ov::element::i8,
                    ov::element::u8,
                    ov::element::u16,
                    ov::element::i16,
                    ov::element::f16,
                    ov::element::i32,
                    ov::element::f32)) {
            DEBUG_LOG("NECopy does not support source precision: ", convertParams.srcPrc.to_string());
            return false;
        }
        if ((convertParams.srcPrc == ov::element::i8 &&
             none_of(convertParams.dstPrc, ov::element::i16, ov::element::i32, ov::element::f16, ov::element::f32)) ||
            (convertParams.srcPrc == ov::element::u8 && none_of(convertParams.dstPrc,
                                                                ov::element::u16,
                                                                ov::element::i16,
                                                                ov::element::i32,
                                                                ov::element::f16,
                                                                ov::element::f32)) ||
            (convertParams.srcPrc == ov::element::u16 &&
             none_of(convertParams.dstPrc, ov::element::u8, ov::element::u32)) ||
            (convertParams.srcPrc == ov::element::i16 &&
             none_of(convertParams.dstPrc, ov::element::i8, ov::element::u8, ov::element::i32)) ||
            (convertParams.srcPrc == ov::element::f16 &&
             none_of(convertParams.dstPrc, ov::element::i8, ov::element::f32, ov::element::i32, ov::element::u8)) ||
            (convertParams.srcPrc == ov::element::i32 &&
             none_of(convertParams.dstPrc, ov::element::i8, ov::element::f16, ov::element::f32, ov::element::u8)) ||
            (convertParams.srcPrc == ov::element::f32 &&
             none_of(convertParams.dstPrc, ov::element::bf16, ov::element::f16, ov::element::i32))) {
            DEBUG_LOG("NECopy does not support passed combination of source and destination precisions. ",
                      "source precision: ",
                      convertParams.srcPrc.to_string(),
                      " destination precsion: ",
                      convertParams.dstPrc.to_string());
            return false;
        }
    }
    return true;
}

}  // namespace ov::intel_cpu
