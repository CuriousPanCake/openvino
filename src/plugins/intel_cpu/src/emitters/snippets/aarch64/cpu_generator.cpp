// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <common/c_types_map.hpp>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/aarch64/jit_conversion_emitters.hpp"
#include "emitters/plugin/aarch64/jit_eltwise_emitters.hpp"
#include "emitters/snippets/aarch64/jit_fill_emitter.hpp"
#include "emitters/snippets/aarch64/jit_gemm_copy_b_emitter.hpp"
#include "emitters/snippets/aarch64/jit_gemm_emitter.hpp"
#include "emitters/snippets/aarch64/jit_horizon_emitters.hpp"
#include "emitters/snippets/aarch64/jit_kernel_emitter.hpp"
#include "emitters/snippets/aarch64/jit_loop_emitters.hpp"
#include "emitters/snippets/aarch64/jit_memory_emitters.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "jit_snippets_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/xor.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/target_machine.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "utils/general_utils.h"

#ifdef SNIPPETS_LIBXSMM_TPP
#    include "emitters/tpp/aarch64/jit_brgemm_emitter.hpp"
#    include "transformations/tpp/common/op/brgemm.hpp"
#endif

namespace ov {

#define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                      \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                      \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         return e_type::get_supported_precisions(n);                                              \
     }}

#define CREATE_CPU_EMITTER(e_type)                                                                \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return std::make_shared<e_type>(h.get(), isa, expr->get_node());                         \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         return e_type::get_supported_precisions(n);                                              \
     }}

#define CREATE_GELU_V7_EMITTER(e_type_erf, e_type_tanh)                                           \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         const auto& n = expr->get_node();                                                        \
         const auto& gelu = ov::as_type_ptr<ov::op::v7::Gelu>(n);                                 \
         if (gelu == nullptr) {                                                                   \
             OPENVINO_THROW("Can't cast to ov::op::v7::Gelu");                                    \
         }                                                                                        \
         const auto approximationMode = gelu->get_approximation_mode();                           \
         if (approximationMode == ov::op::GeluApproximationMode::ERF) {                           \
             return std::make_shared<e_type_erf>(h.get(), isa, n);                                \
         }                                                                                        \
         if (approximationMode == ov::op::GeluApproximationMode::TANH) {                          \
             return std::make_shared<e_type_tanh>(h.get(), isa, n);                               \
         }                                                                                        \
         OPENVINO_THROW("Unsupported Gelu approximation mode");                                   \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         const auto& gelu = ov::as_type_ptr<ov::op::v7::Gelu>(n);                                 \
         if (gelu == nullptr) {                                                                   \
             OPENVINO_THROW("Can't cast to ov::op::v7::Gelu");                                    \
         }                                                                                        \
         const auto approximationMode = gelu->get_approximation_mode();                           \
         if (approximationMode == ov::op::GeluApproximationMode::ERF) {                           \
             return e_type_erf::get_supported_precisions(n);                                      \
         }                                                                                        \
         if (approximationMode == ov::op::GeluApproximationMode::TANH) {                          \
             return e_type_tanh::get_supported_precisions(n);                                     \
         }                                                                                        \
         OPENVINO_THROW("Unsupported Gelu approximation mode");                                   \
     }}

#define CREATE_ROUND_V5_EMITTER(e_type_from_zero, e_type_even)                                    \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         const auto& n = expr->get_node();                                                        \
         const auto& round = ov::as_type_ptr<ov::op::v5::Round>(n);                               \
         if (round == nullptr) {                                                                  \
             OPENVINO_THROW("Can't cast to ov::op::v5::Round");                                   \
         }                                                                                        \
         const auto roundingMode = round->get_mode();                                             \
         if (roundingMode == ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO) {                 \
             return std::make_shared<e_type_from_zero>(h.get(), isa, n);                          \
         }                                                                                        \
         if (roundingMode == ov::op::v5::Round::RoundMode::HALF_TO_EVEN) {                        \
             return std::make_shared<e_type_even>(h.get(), isa, n);                               \
         }                                                                                        \
         OPENVINO_THROW("Unsupported Round mode");                                                \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         const auto& round = ov::as_type_ptr<ov::op::v5::Round>(n);                               \
         if (round == nullptr) {                                                                  \
             OPENVINO_THROW("Can't cast to ov::op::v5::Round");                                   \
         }                                                                                        \
         if (round->get_mode() == ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO) {            \
             return e_type_from_zero::get_supported_precisions(n);                                \
         }                                                                                        \
         if (round->get_mode() == ov::op::v5::Round::RoundMode::HALF_TO_EVEN) {                   \
             return e_type_even::get_supported_precisions(n);                                     \
         }                                                                                        \
         OPENVINO_THROW("Unsupported Round mode");                                                \
     }}

#define CREATE_UNDEFINED_EMITTER(supported_precisions)                                                         \
    {[]([[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return nullptr;                                                                                       \
     },                                                                                                        \
     []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {         \
         return supported_precisions;                                                                          \
     }}

class jit_snippet : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() override = default;

    jit_snippet() = default;

    void generate() override {}
};

namespace intel_cpu::aarch64 {

CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h)
    : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr cache)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>(cache)),
      h(new jit_snippet()),
      isa(host_isa),
      compiled_kernel_cache(std::move(cache)) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::Buffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::Buffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::Reshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::Reorder::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_broadcast_move_emitter);
    jitters[snippets::op::ConvertTruncation::get_type_info_static()] =
        CREATE_CPU_EMITTER(jit_convert_truncation_emitter);
    jitters[snippets::op::ConvertSaturation::get_type_info_static()] =
        CREATE_CPU_EMITTER(jit_convert_saturation_emitter);

    // memory access
    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_memory_emitter);
    jitters[snippets::op::LoadReorder::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_broadcast_emitter);
    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_store_memory_emitter);

    // ternary
    jitters[op::v1::Select::get_type_info_static()] = CREATE_CPU_EMITTER(jit_select_emitter);
    jitters[intel_cpu::FusedMulAdd::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mul_add_emitter);

    // binary
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(jit_add_emitter);
    jitters[op::v1::Divide::get_type_info_static()] = CREATE_CPU_EMITTER(jit_divide_emitter);
    jitters[op::v1::Maximum::get_type_info_static()] = CREATE_CPU_EMITTER(jit_maximum_emitter);
    jitters[op::v1::Minimum::get_type_info_static()] = CREATE_CPU_EMITTER(jit_minimum_emitter);
    jitters[op::v1::Mod::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mod_emitter);
    jitters[op::v1::Multiply::get_type_info_static()] = CREATE_CPU_EMITTER(jit_multiply_emitter);
    jitters[snippets::op::PowerStatic::get_type_info_static()] = CREATE_CPU_EMITTER(jit_power_static_emitter);
    jitters[op::v1::Power::get_type_info_static()] = CREATE_CPU_EMITTER(jit_power_dynamic_emitter);
    jitters[op::v0::SquaredDifference::get_type_info_static()] = CREATE_CPU_EMITTER(jit_squared_difference_emitter);
    jitters[op::v1::Subtract::get_type_info_static()] = CREATE_CPU_EMITTER(jit_subtract_emitter);
    jitters[op::v0::Xor::get_type_info_static()] = CREATE_CPU_EMITTER(jit_logical_xor_emitter);

    // Comparison ops
    jitters[op::v1::Equal::get_type_info_static()] = CREATE_CPU_EMITTER(jit_equal_emitter);
    jitters[op::v1::Greater::get_type_info_static()] = CREATE_CPU_EMITTER(jit_greater_emitter);
    jitters[op::v1::GreaterEqual::get_type_info_static()] = CREATE_CPU_EMITTER(jit_greater_equal_emitter);
    jitters[op::v1::Less::get_type_info_static()] = CREATE_CPU_EMITTER(jit_less_emitter);
    jitters[op::v1::LessEqual::get_type_info_static()] = CREATE_CPU_EMITTER(jit_less_equal_emitter);
    jitters[op::v1::NotEqual::get_type_info_static()] = CREATE_CPU_EMITTER(jit_not_equal_emitter);

    // Logical ops
    jitters[op::v1::LogicalAnd::get_type_info_static()] = CREATE_CPU_EMITTER(jit_logical_and_emitter);
    jitters[op::v1::LogicalOr::get_type_info_static()] = CREATE_CPU_EMITTER(jit_logical_or_emitter);
    jitters[op::v1::LogicalNot::get_type_info_static()] = CREATE_CPU_EMITTER(jit_logical_not_emitter);
    jitters[op::v1::LogicalXor::get_type_info_static()] = CREATE_CPU_EMITTER(jit_logical_xor_emitter);

    // unary
    jitters[ov::op::v0::Abs::get_type_info_static()] = CREATE_CPU_EMITTER(jit_abs_emitter);
    jitters[ov::op::v0::Ceiling::get_type_info_static()] = CREATE_CPU_EMITTER(jit_ceiling_emitter);
    jitters[ov::op::v0::Clamp::get_type_info_static()] = CREATE_CPU_EMITTER(jit_clamp_emitter);
    jitters[ov::op::v0::Elu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_elu_emitter);
    jitters[ov::op::v0::Erf::get_type_info_static()] = CREATE_CPU_EMITTER(jit_erf_emitter);
    jitters[ov::op::v0::Exp::get_type_info_static()] = CREATE_CPU_EMITTER(jit_exp_emitter);
    jitters[ov::op::v0::Floor::get_type_info_static()] = CREATE_CPU_EMITTER(jit_floor_emitter);
    jitters[ov::op::v1::FloorMod::get_type_info_static()] = CREATE_CPU_EMITTER(jit_floor_mod_emitter);
    jitters[ov::op::v0::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_gelu_erf_emitter);
    jitters[ov::op::v7::Gelu::get_type_info_static()] =
        CREATE_GELU_V7_EMITTER(jit_gelu_erf_emitter, jit_gelu_tanh_emitter);
    jitters[ov::op::v4::HSwish::get_type_info_static()] = CREATE_CPU_EMITTER(jit_hswish_emitter);
    jitters[ov::op::v4::Mish::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mish_emitter);
    jitters[ov::op::v0::Negative::get_type_info_static()] = CREATE_CPU_EMITTER(jit_negative_emitter);
    jitters[ov::op::v0::PRelu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_prelu_emitter);
    jitters[ov::op::v0::Relu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_relu_emitter);
    jitters[ov::op::v5::Round::get_type_info_static()] =
        CREATE_ROUND_V5_EMITTER(jit_round_half_away_from_zero_emitter, jit_round_half_to_even_emitter);
    jitters[ov::op::v0::Sigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(jit_sigmoid_emitter);
    jitters[ov::op::v0::Sqrt::get_type_info_static()] = CREATE_CPU_EMITTER(jit_sqrt_emitter);
    jitters[ov::intel_cpu::SwishNode::get_type_info_static()] = CREATE_CPU_EMITTER(jit_swish_emitter);
    jitters[ov::op::v0::Tanh::get_type_info_static()] = CREATE_CPU_EMITTER(jit_tanh_emitter);
    jitters[ov::intel_cpu::aarch64::GemmCPU::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(jit_gemm_emitter, configurator->get_kernel_executor_table());
    jitters[ov::intel_cpu::aarch64::GemmCopyB::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(jit_gemm_copy_b_emitter, configurator->get_kernel_executor_table());
#ifdef SNIPPETS_LIBXSMM_TPP
    // brgemm
    jitters[ov::intel_cpu::tpp::op::BrgemmTPP::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(jit_brgemm_emitter, configurator->get_kernel_executor_table(), compiled_kernel_cache);
#endif

    // reductions
    jitters[ov::snippets::op::ReduceMax::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});
    jitters[ov::snippets::op::ReduceSum::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});
    jitters[ov::snippets::op::HorizonMax::get_type_info_static()] = CREATE_CPU_EMITTER(jit_horizon_max_emitter);
    jitters[ov::snippets::op::HorizonSum::get_type_info_static()] = CREATE_CPU_EMITTER(jit_horizon_sum_emitter);
    // control flow
    jitters[snippets::op::KernelStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_static_emitter);
    jitters[snippets::op::KernelDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_dynamic_emitter);
    jitters[snippets::op::LoopBegin::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_begin_emitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_end_emitter);

    // others
    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_scalar_emitter);
    jitters[snippets::op::Fill::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_fill_emitter);
}

std::shared_ptr<snippets::TargetMachine> CPUTargetMachine::clone() const {
    const auto cloned = std::make_shared<CPUTargetMachine>(isa, compiled_kernel_cache);
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
    return cloned;
}

bool CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
}

snippets::CompiledSnippetPtr CPUTargetMachine::get_snippet() {
    OPENVINO_ASSERT(h->create_kernel() == dnnl::impl::status::success, "Failed to create jit_kernel in get_snippet()");

    const auto& result =
        std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h = std::make_unique<jit_snippet>();
    return result;
}

size_t CPUTargetMachine::get_lanes() const {
    switch (isa) {
    case dnnl::impl::cpu::aarch64::asimd:
        return dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::vlen / sizeof(float);
    default:
        OPENVINO_THROW("unknown isa ", isa);
    }
}

std::vector<snippets::Reg> CPUTargetMachine::get_abi_arg_regs() const {
    using namespace dnnl::impl::cpu::aarch64;
    std::vector<snippets::Reg> res;
    for (const auto& r :
         {abi_param1, abi_param2, abi_param3, abi_param4, abi_param5, abi_param6, abi_param7, abi_param8}) {
        res.emplace_back(snippets::RegType::gpr, r.getIdx());
    }
    return res;
}

std::vector<snippets::Reg> CPUTargetMachine::get_gp_reg_pool() const {
    using Xbyak_aarch64::Operand;
    const auto num_gp_regs = 32;
    std::vector<snippets::Reg> reg_pool;
    for (size_t i = 0; i < num_gp_regs; i++) {
        // Note: more details on the usage of reserved registers in aarch64/jit_kernel_emitter.cpp
        if (none_of(i, Operand::SP, Operand::X18, Operand::X23, Operand::X24, Operand::X28, Operand::X29)) {
            reg_pool.emplace_back(snippets::RegType::gpr, i);
        }
    }
    return reg_pool;
}

std::vector<snippets::Reg> CPUTargetMachine::get_vec_reg_pool() const {
    const auto num_vec_regs = [this]() {
        switch (isa) {
        case dnnl::impl::cpu::aarch64::asimd:
            return dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::n_vregs;
        default:
            OPENVINO_THROW("unknown isa ", isa);
        }
    }();
    std::vector<snippets::Reg> reg_pool;
    reg_pool.reserve(num_vec_regs);
    for (int i = 0; i < num_vec_regs; i++) {
        reg_pool.emplace_back(snippets::RegType::vec, static_cast<size_t>(i));
    }
    return reg_pool;
}

dnnl::impl::cpu::aarch64::cpu_isa_t CPUTargetMachine::get_isa() const {
    return isa;
}

CPUGenerator::CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa_, ov::intel_cpu::MultiCacheWeakPtr cache)
    : Generator(std::make_shared<CPUTargetMachine>(isa_, std::move(cache))) {}
CPUGenerator::CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target) : Generator(target) {}

std::shared_ptr<snippets::Generator> CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine,
                    "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine);
}

ov::snippets::RegType CPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    const auto op = out.get_node_shared_ptr();
    if (is_type_any_of<intel_cpu::aarch64::GemmCPU, intel_cpu::aarch64::GemmCopyB>(op)) {
        return ov::snippets::RegType::gpr;
    }
    if (ov::is_type_any_of<intel_cpu::FusedMulAdd, intel_cpu::SwishNode>(op)) {
        return ov::snippets::RegType::vec;
    }
    return ov::snippets::RegType::undefined;
}

bool CPUGenerator::uses_precompiled_kernel([[maybe_unused]] const std::shared_ptr<snippets::Emitter>& e) const {
    return false;
}

}  // namespace intel_cpu::aarch64

}  // namespace ov
