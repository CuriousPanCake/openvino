// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/extract_first_sdpa.hpp"

#include <unordered_set>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::ExtractFirstSDPA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(ExtractFirstSDPA);

    // Step 1: Find the first SDPA in topological order
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> first_sdpa = nullptr;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
            first_sdpa = sdpa;
            break;
        }
    }

    if (!first_sdpa) {
        // No SDPA found, nothing to do
        return false;
    }

    // Step 2: Save original result name (use first result's name if available)
    std::string original_result_name;
    ov::TensorNames original_result_tensor_name;
    auto original_results = model->get_results();
    if (!original_results.empty()) {
        original_result_name = original_results[0]->get_friendly_name();
        original_result_tensor_name = original_results[0]->get_output_tensor(0).get_names();
    }

    // Step 3: Remove all existing results
    for (const auto& result : original_results) {
        model->remove_result(result);
    }

    // Step 4: Create a new Result connected to the first SDPA's output, preserving original name
    auto new_result = std::make_shared<ov::op::v0::Result>(first_sdpa->output(0));
    if (!original_result_name.empty()) {
        new_result->set_friendly_name(original_result_name);
    } else {
        new_result->set_friendly_name("sdpa_output");
    }
    new_result->get_output_tensor(0).set_names(original_result_tensor_name);
    model->add_results({new_result});

    // Step 5: Collect all nodes reachable from the new result (needed for SDPA computation)
    std::unordered_set<ov::Node*> needed_nodes;
    ov::traverse_nodes({new_result}, [&needed_nodes](const std::shared_ptr<ov::Node>& node) {
        needed_nodes.insert(node.get());
    });

    // Step 6: Identify which variables are used by the first SDPA (for KV-cache statefulness)
    std::unordered_set<std::string> needed_variable_ids;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(op)) {
            if (needed_nodes.find(op.get()) != needed_nodes.end()) {
                needed_variable_ids.insert(read_value->get_variable_id());
            }
        }
    }

    // Step 7: Find Assigns for needed variables and add them to needed_nodes
    // We need to traverse from Assigns to include all nodes that feed into them
    std::vector<std::shared_ptr<ov::op::util::AssignBase>> needed_assigns;
    auto sinks = model->get_sinks();
    for (const auto& sink : sinks) {
        if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
            if (needed_variable_ids.count(assign->get_variable_id())) {
                needed_assigns.push_back(assign);
            }
        }
    }

    // Traverse from needed Assigns to collect all nodes they depend on
    for (const auto& assign : needed_assigns) {
        ov::traverse_nodes({assign}, [&needed_nodes](const std::shared_ptr<ov::Node>& node) {
            needed_nodes.insert(node.get());
        });
    }

    // Step 8: Remove sinks (Assigns) that are NOT needed
    for (const auto& sink : sinks) {
        if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
            if (!needed_variable_ids.count(assign->get_variable_id())) {
                model->remove_sink(sink);
            }
        }
    }

    // Step 9: Find and remove unused parameters
    auto params = model->get_parameters();
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params_to_remove;
    for (const auto& param : params) {
        if (needed_nodes.find(param.get()) == needed_nodes.end()) {
            params_to_remove.push_back(param);
        }
    }
    for (const auto& param : params_to_remove) {
        model->remove_parameter(param);
    }

    // Step 10: Remove unused variables
    auto variables = model->get_variables();
    std::vector<std::shared_ptr<ov::op::util::Variable>> vars_to_remove;
    for (const auto& var : variables) {
        if (!needed_variable_ids.count(var->get_info().variable_id)) {
            vars_to_remove.push_back(var);
        }
    }
    for (const auto& var : vars_to_remove) {
        model->remove_variable(var);
    }

    // Step 11: Validate the model
    model->validate_nodes_and_infer_types();

    return true;
}
