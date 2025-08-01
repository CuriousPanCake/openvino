// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_eltwise.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Dimensions.h>
#include <arm_compute/core/Rounding.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPReluLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>
#include <arm_compute/runtime/Tensor.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace arm_compute;

inline VectorDims reshape_sizes(VectorDims dims) {
    static constexpr size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;

    if (dims.size() < MAX_NUM_SHAPE) {
        return dims;
    }

    VectorDims result_dims(MAX_NUM_SHAPE - 1);

    for (size_t i = 0; i < MAX_NUM_SHAPE - 1; i++) {
        result_dims[i] = dims[i];
    }
    for (size_t i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
        result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
    }

    return result_dims;
}

inline void log_unsupported_prec(const std::vector<MemoryDescPtr>& srcDescs,
                                 const std::vector<MemoryDescPtr>& dstDescs,
                                 const Algorithm eltwiseAlgorithm) {
    std::string srcPrec;
    for (const auto& srcDesc : srcDescs) {
        srcPrec += srcDesc->getPrecision().to_string() + " ";
    }
    DEBUG_LOG(algToString(eltwiseAlgorithm),
              ": provided combination of src precisions: [",
              srcPrec,
              "] and dst precision: ",
              dstDescs[0]->getPrecision().to_string(),
              " is not supported");
}

bool AclEltwiseExecutor::supports(const EltwiseConfig& config) {
    std::vector<MemoryDescPtr> srcDescs(config.descs.size() - 1);
    std::vector<MemoryDescPtr> dstDescs{config.descs.at(ARG_DST)};

    for (const auto& [argId, desc] : config.descs) {
        if (argId == ARG_DST) {
            continue;
        }

        srcDescs[argId - ARG_SRC] = desc;
    }

    auto checkPrecision = [&srcDescs, &dstDescs](std::vector<ov::element::Type> srcVecPrc,
                                                 ov::element::Type dstPrc) -> bool {
        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (srcDescs[i]->getPrecision() != srcVecPrc[i]) {
                return false;
            }
        }
        if (dstDescs[0]->getPrecision() != dstPrc) {
            return false;
        }
        return true;
    };

    const auto& eltwiseAttrs = config.attrs;

    switch (eltwiseAttrs.data.algo) {
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseRelu:
#ifdef OPENVINO_ARCH_ARM64
    case Algorithm::EltwiseGeluErf:
#endif
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwisePrelu:
    case Algorithm::EltwiseHswish:
        if (!(checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseAbs:
    case Algorithm::EltwiseExp:
    case Algorithm::EltwiseLog:
        if (!(checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
              checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseMaximum:
    case Algorithm::EltwiseMinimum:
    case Algorithm::EltwiseSquaredDifference:
        if (!(checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
              checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
              checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
        if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
              checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
              checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
              checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseMultiply:
        if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
              checkPrecision({ov::element::u8, ov::element::u8}, ov::element::i16) ||
              checkPrecision({ov::element::u8, ov::element::i16}, ov::element::i16) ||
              checkPrecision({ov::element::i16, ov::element::u8}, ov::element::i16) ||
              checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
              checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    // ACL supports only U8 precision on output for comparison operations
    case Algorithm::EltwiseEqual:
    case Algorithm::EltwiseNotEqual:
    case Algorithm::EltwiseGreater:
    case Algorithm::EltwiseGreaterEqual:
    case Algorithm::EltwiseLess:
    case Algorithm::EltwiseLessEqual:
        if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
              checkPrecision({ov::element::i16, ov::element::i16}, ov::element::u8) ||
              checkPrecision({ov::element::i32, ov::element::i32}, ov::element::u8) ||
              checkPrecision({ov::element::f16, ov::element::f16}, ov::element::u8) ||
              checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    default:
        DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.data.algo), " is not supported");
        return false;
    }

    for (const auto& srcDesc : srcDescs) {
        if (getAclDataLayoutByMemoryDesc(srcDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("src descriptor layout is unsupported by ACL: ", srcDesc->serializeFormat());
            return false;
        }
    }
    for (const auto& dstDesc : dstDescs) {
        if (getAclDataLayoutByMemoryDesc(dstDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("dst descriptor layout is unsupported by ACL: ", dstDesc->serializeFormat());
            return false;
        }
    }

    return true;
}

AclEltwiseExecutor::AclEltwiseExecutor(EltwiseAttrs attrs,
                                       [[maybe_unused]] const MemoryArgs& memory,
                                       [[maybe_unused]] const ExecutorContext::CPtr& context)
    : aclEltwiseAttrs(std::move(attrs)) {}

bool AclEltwiseExecutor::init(const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) {
    auto postOps = aclEltwiseAttrs.postOps;

    if (!postOps.empty()) {
        return false;
    }

    std::vector<arm_compute::TensorShape> srcVecDims(srcDescs.size()), dstVecDims(dstDescs.size());
    std::vector<arm_compute::DataLayout> srcDataLayout(srcDescs.size()), dstDataLayout(dstDescs.size());
    std::vector<arm_compute::TensorInfo> srcTensorsInfo(srcDescs.size()), dstTensorsInfo(dstDescs.size());
    srcTensors = std::vector<arm_compute::Tensor>(srcDescs.size());
    dstTensors = std::vector<arm_compute::Tensor>(dstDescs.size());

    for (size_t i = 0; i < srcVecDims.size(); i++) {
        srcVecDims[i] = shapeCast(reshape_sizes(srcDescs[i]->getShape().getDims()));
    }
    for (size_t i = 0; i < dstVecDims.size(); i++) {
        dstVecDims[i] = shapeCast(reshape_sizes(dstDescs[i]->getShape().getDims()));
    }

    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcDataLayout[i] = getAclDataLayoutByMemoryDesc(srcDescs[i]);
        if (srcDataLayout[i] == arm_compute::DataLayout::UNKNOWN) {
            return false;
        }
    }
    for (size_t i = 0; i < dstDescs.size(); i++) {
        dstDataLayout[i] = getAclDataLayoutByMemoryDesc(dstDescs[i]);
        if (dstDataLayout[i] == arm_compute::DataLayout::UNKNOWN) {
            return false;
        }
    }

    if (srcDescs.size() == 2 && srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
        srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
        srcDescs[0]->getShape().getDims() != srcDescs[1]->getShape().getDims()) {
        if (srcVecDims[0].num_dimensions() < 5) {
            srcDataLayout[0] = srcDataLayout[1] = dstDataLayout[0] = DataLayout::NCHW;
        } else {
            srcDataLayout[0] = srcDataLayout[1] = dstDataLayout[0] = DataLayout::NCDHW;
        }
        changeLayoutToNH_C({srcVecDims.data(), &(srcVecDims[1]), dstVecDims.data()});
    }

    for (size_t i = 0; i < srcVecDims.size(); i++) {
        srcTensorsInfo[i] =
            TensorInfo(srcVecDims[i], 1, precisionToAclDataType(srcDescs[i]->getPrecision()), srcDataLayout[i]);
        srcTensors[i].allocator()->init(srcTensorsInfo[i]);
    }

    for (size_t i = 0; i < dstVecDims.size(); i++) {
        dstTensorsInfo[i] =
            TensorInfo(dstVecDims[i], 1, precisionToAclDataType(dstDescs[i]->getPrecision()), dstDataLayout[i]);
        dstTensors[i].allocator()->init(dstTensorsInfo[i]);
    }

    std::function<std::unique_ptr<IFunction>(void)> exec_func;
    switch (aclEltwiseAttrs.data.algo) {
    case Algorithm::EltwiseAdd:
        if (!NEArithmeticAddition::validate(srcTensorsInfo.data(),
                                            &srcTensorsInfo[1],
                                            dstTensorsInfo.data(),
                                            ConvertPolicy::SATURATE)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEArithmeticAddition>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ConvertPolicy::SATURATE);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseMultiply:
        if (!NEPixelWiseMultiplication::validate(srcTensorsInfo.data(),
                                                 &srcTensorsInfo[1],
                                                 dstTensorsInfo.data(),
                                                 1.0f,
                                                 ConvertPolicy::SATURATE,
                                                 RoundingPolicy::TO_ZERO)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEPixelWiseMultiplication>();
            acl_op->configure(srcTensors.data(),
                              &srcTensors[1],
                              dstTensors.data(),
                              1.0f,
                              ConvertPolicy::SATURATE,
                              RoundingPolicy::TO_ZERO);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseSubtract:
        if (!NEArithmeticSubtraction::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ConvertPolicy::SATURATE)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEArithmeticSubtraction>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ConvertPolicy::SATURATE);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseDivide:
        if (!NEElementwiseDivision::validate(srcTensorsInfo.data(), &srcTensorsInfo[1], dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseDivision>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseMaximum:
        if (!NEElementwiseMax::validate(srcTensorsInfo.data(), &srcTensorsInfo[1], dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseMax>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseMinimum:
        if (!NEElementwiseMin::validate(srcTensorsInfo.data(), &srcTensorsInfo[1], dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseMin>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseSquaredDifference:
        if (!NEElementwiseSquaredDiff::validate(srcTensorsInfo.data(), &srcTensorsInfo[1], dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseSquaredDiff>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseEqual:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::Equal)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::Equal);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseNotEqual:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::NotEqual)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::NotEqual);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseGreater:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::Greater)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::Greater);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseGreaterEqual:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::GreaterEqual)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::GreaterEqual);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseLess:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::Less)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::Less);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseLessEqual:
        if (!NEElementwiseComparison::validate(srcTensorsInfo.data(),
                                               &srcTensorsInfo[1],
                                               dstTensorsInfo.data(),
                                               ComparisonOperation::LessEqual)) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEElementwiseComparison>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data(), ComparisonOperation::LessEqual);
            return acl_op;
        };
        break;
    case Algorithm::EltwiseAbs:
        if (!NEAbsLayer::validate(srcTensorsInfo.data(), dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEAbsLayer>();
            acl_op->configure(srcTensors.data(), dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseExp:
        if (!NEExpLayer::validate(srcTensorsInfo.data(), dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEExpLayer>();
            acl_op->configure(srcTensors.data(), dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwisePrelu:
        if (!NEPReluLayer::validate(srcTensorsInfo.data(), &srcTensorsInfo[1], dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEPReluLayer>();
            acl_op->configure(srcTensors.data(), &srcTensors[1], dstTensors.data());
            return acl_op;
        };
        break;
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish:
        if (!NEActivationLayer::validate(srcTensorsInfo.data(),
                                         dstTensorsInfo.data(),
                                         getActivationLayerInfo(aclEltwiseAttrs.data.algo,
                                                                aclEltwiseAttrs.data.alpha,
                                                                aclEltwiseAttrs.data.beta,
                                                                aclEltwiseAttrs.data.gamma))) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NEActivationLayer>();
            acl_op->configure(srcTensors.data(),
                              dstTensors.data(),
                              getActivationLayerInfo(aclEltwiseAttrs.data.algo,
                                                     aclEltwiseAttrs.data.alpha,
                                                     aclEltwiseAttrs.data.beta,
                                                     aclEltwiseAttrs.data.gamma));
            return acl_op;
        };
        break;
    case Algorithm::EltwiseLog:
        if (!NELogLayer::validate(srcTensorsInfo.data(), dstTensorsInfo.data())) {
            return false;
        }
        exec_func = [this]() -> std::unique_ptr<IFunction> {
            auto acl_op = std::make_unique<NELogLayer>();
            acl_op->configure(srcTensors.data(), dstTensors.data());
            return acl_op;
        };
        break;
    default:
        OPENVINO_THROW("Unsupported operation type for ACL Eltwise executor: ",
                       static_cast<int>(aclEltwiseAttrs.data.algo));
    }

    configureThreadSafe([&] {
        ifunc = exec_func();
    });
    return true;
}

bool AclEltwiseExecutor::update(const MemoryArgs& memory) {
    std::vector<MemoryDescPtr> srcDescs(memory.size() - 1);
    std::vector<MemoryDescPtr> dstDescs{memory.at(ARG_DST)->getDescPtr()};

    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        srcDescs[argId - ARG_SRC] = mem->getDescPtr();
    }

    if (!init(srcDescs, dstDescs)) {
        return false;
    }

    return true;
}

void AclEltwiseExecutor::execute(const MemoryArgs& memory) {
    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        srcTensors[argId - ARG_SRC].allocator()->import_memory(mem->getData());
    }

    dstTensors[0].allocator()->import_memory(memory.at(ARG_DST)->getData());

    ifunc->run();

    for (auto& srcTensor : srcTensors) {
        srcTensor.allocator()->free();
    }
    for (auto& dstTensor : dstTensors) {
        dstTensor.allocator()->free();
    }
}

}  // namespace ov::intel_cpu
