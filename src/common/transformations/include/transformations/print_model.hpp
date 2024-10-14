// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/util/multi_subgraph_base.hpp"

#include "transformations/utils/gen_pattern.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov {
namespace pass {

namespace detail {

// to_code convert value into literal/constexpr/initializer_list/factory_calls in C++ source code
inline std::string to_code(bool value) {
    return value ? "true" : "false";
}

inline std::string to_code(const std::string& value) {
    return std::string("\"") + value + "\"";
}

inline std::string to_code(const element::Type& value) {
    return std::string("ov::element::") + value.to_string();
}

inline std::string to_code(const ov::Shape& shape) {
    return "\"" + shape.to_string().substr(1, shape.to_string().size() - 2) + "\"";
}

inline std::string to_code(const ov::PartialShape& shape) {
    return "\"" + shape.to_string().substr(1, shape.to_string().size() - 2) + "\"";
}

inline std::string to_code(int value) {
    if (INT_MAX == value) {
        return "INT_MAX";
    }
    if (INT_MIN == value) {
        return "INT_MIN";
    }
    return std::to_string(value);
}
inline std::string to_code(int64_t value) {
    if (LLONG_MAX == value) {
        return "LLONG_MAX";
    }
    if (LLONG_MIN == value) {
        return "LLONG_MIN";
    }
    const char* suffix = "LL";
    if (value == static_cast<int64_t>(static_cast<int>(value))) {
        // save suffix since most values can be expressed as int
        // this produces more readable code
        suffix = "";
    }
    return std::to_string(value) + suffix;
}
inline std::string to_code(uint64_t value) {
    if (ULLONG_MAX == value) {
        return "ULLONG_MAX";
    }
    const char* suffix = "uLL";
    if (value == static_cast<uint64_t>(static_cast<int>(value))) {
        // save suffix since most values can be expressed as int
        // this produces more readable code
        suffix = "";
    }
    return std::to_string(value) + suffix;
}
inline std::string to_code(int8_t value) {
    return std::to_string(static_cast<int>(value));
}
inline std::string to_code(uint8_t value) {
    return std::to_string(static_cast<int>(value));
}

template <typename T>
std::string to_code_float(T value) {
    if (std::isnan(value)) {
        return "NAN";
    } else if (std::isinf(value)) {
        return (value > 0 ? "INFINITY" : "-INFINITY");
    } else if (value == FLT_MIN) {
        return "FLT_MIN";
    } else if (value == -FLT_MIN) {
        return "-FLT_MIN";
    } else if (value == FLT_MAX) {
        return "FLT_MAX";
    } else if (value == -FLT_MAX) {
        return "-FLT_MAX";
    }
    auto strv = std::to_string(value);
    if (strv.find(".") == std::string::npos && strv.find("e") == std::string::npos)
        strv += ".0";
    if (std::is_same<T, float>::value)
        strv += "f";
    return strv;
}

inline std::string to_code(float value) {
    return to_code_float(value);
}
inline std::string to_code(double value) {
    return to_code_float(value);
}
template <typename T>
std::string to_code(const std::vector<T>& values, bool no_braces = false, size_t max_elements = 5) {
    std::string string;
    if (!no_braces)
        string += "{";
    auto sep = "";
    for (size_t i = 0; i < values.size() && i < max_elements; ++i) {
        string += sep;
        string += to_code(values[i]);
        sep = ", ";
    }
    if (values.size() > max_elements) {
        string += "... (" + std::to_string(values.size()) + " in total)";
    }
    if (!no_braces)
        string += "}";
    return string;
}

template <typename T = void>
std::string to_code(std::shared_ptr<ov::op::v0::Constant> constop, bool force_braces = false) {
    bool no_braces = (constop->get_shape().size() == 0) && (!force_braces);
    auto ele_type = constop->get_element_type();
    if (ele_type == element::Type_t::f32) {
        return to_code(constop->get_vector<float>(), no_braces);
    } else if (ele_type == element::Type_t::i8) {
        return to_code(constop->get_vector<int8_t>(), no_braces);
    } else if (ele_type == element::Type_t::u8) {
        return to_code(constop->get_vector<uint8_t>(), no_braces);
    } else if (ele_type == element::Type_t::i32) {
        return to_code(constop->get_vector<int32_t>(), no_braces);
    } else if (ele_type == element::Type_t::i64) {
        return to_code(constop->get_vector<int64_t>(), no_braces);
    }

    // general case
    std::stringstream ss;
    if (!no_braces)
        ss << "{";
    auto ele_size = shape_size(constop->get_shape());
    if (ele_size < 5) {
        const char* sep = "";
        for (auto v : constop->get_value_strings()) {
            ss << sep << v;
            sep = ", ";
        }
    } else {
        ss << " ...";
    }
    if (!no_braces)
        ss << "}";
    return ss.str();
}

static void foo() {
    auto Constant_49314_pattern = pattern::wrap_type<opset1::Constant>();
    auto Multiply_49316_pattern = pattern::wrap_type<opset1::Multiply>({pattern::any_input(), Constant_49314_pattern});
    auto Gather_49339_pattern = pattern::wrap_type<opset8::Gather>({pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto Concat_49361_pattern = pattern::wrap_type<opset1::Concat>({Gather_49339_pattern, pattern::any_input()});
    auto ScaledDotProductAttention_49408_pattern = pattern::wrap_type<opset13::ScaledDotProductAttention>({Multiply_49316_pattern, Concat_49361_pattern, pattern::any_input(), pattern::any_input()});
    auto Assign_50025_pattern = pattern::wrap_type<opset6::Assign>({Concat_49361_pattern});
    auto Add_50153_pattern = pattern::wrap_type<opset1::Add>({pattern::any_input(), pattern::any_input()});
    auto MatMul_50155_pattern = pattern::wrap_type<opset1::MatMul>({Add_50153_pattern, pattern::any_input()});
    auto Result_50157_pattern = pattern::wrap_type<opset1::Result>({MatMul_50155_pattern});
}

class OstreamAttributeVisitor : public ov::AttributeVisitor {
    std::ostream& os;
    const char* sep = "";

public:
    OstreamAttributeVisitor(std::ostream& os) : os(os) {}

    void append_attribute(const std::string& name, const std::string& value) {
        os << sep << "{\"" << name << "\", " << value << "}";
        sep = ", ";
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& strset = a->get();
            std::vector<std::string> values(strset.begin(), strset.end());
            append_attribute(name, to_code(values));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            append_attribute(name, to_code(a->get()));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            append_attribute(name, a->get().to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            const auto& vinfo = a->get()->get_info();
            std::string str = "ov::op::util::VariableInfo{{" + to_code(vinfo.data_shape) + "}, " + to_code(vinfo.data_type) + ", \"" + vinfo.variable_id + "\"}";
            append_attribute(name, str);
        } else {
            append_attribute(name, "?");
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        append_attribute(name, "Model");
    }
};

using NodePtr = std::shared_ptr<ov::Node>;

static void recurse_down_dfs(const NodePtr& node, std::unordered_set<NodePtr>& path_nodes, const std::vector<std::string>& end_nodes) {
    if (path_nodes.find(node) != path_nodes.end()) { // the node has already been processed
        return;
    }

    path_nodes.insert(node);

    bool is_ending_node = false;
    for (auto& name : end_nodes) { //TODO: change to std::any_of
        if (node->get_name() == name) {
            is_ending_node = true;
            break;
        }
    }

    if (std::dynamic_pointer_cast<ov::op::v0::Result>(node) || // What other types we consider as end nodes?
        std::dynamic_pointer_cast<ov::op::v3::Assign>(node) ||
        is_ending_node) {
            return;
    }

    for (auto& output : node->outputs()) {
        for (auto& input : output.get_target_inputs()) {
            recurse_down_dfs(input.get_node()->shared_from_this(), path_nodes, end_nodes);
        }
    }
}

static void dump_wrap_type(std::ostream& os,
                           const std::shared_ptr<ov::Model>& model,
                           const std::vector<std::string>& start_nodes,
                           const std::vector<std::string>& end_nodes) {
    std::unordered_set<NodePtr> path_nodes;

    // Find nodes that we would include in our paths and write them to path_nodes
    for (auto& op : model->get_ordered_ops()) {
        for (auto& start_node_name : start_nodes) {
            if (op->get_name() == start_node_name) { // found a node with a given name, we can start processing it
                recurse_down_dfs(op, path_nodes, end_nodes);
            }
        }
    }

    std::unordered_set<NodePtr> printed;

    for (auto& op : model->get_ordered_ops()) {
        if (path_nodes.find(op) != path_nodes.end()) {
            if (auto multi_subgraph = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
                for (size_t i = 0; i < multi_subgraph->get_internal_subgraphs_size(); ++i) {
                    os << "// MultiSubGraphOp " << multi_subgraph->get_name() << "[" << i << "]" << std::endl; // TODO: think if the comment is required here
                    dump_wrap_type(os, multi_subgraph->get_function(i), {}, {});
                }
            } else {
                //TODO: think if there's a need for special hadling of some nodes like Parameter or Constant. For now it doesn't seem like that
                auto type = op->get_type_info().get_version() + "::" + std::string(op->get_type_name());
                auto print_node = "auto " + op->get_name() + "_pattern = pattern::wrap_type<" + type + ">(";
                for (size_t i = 0; i < op->get_input_size(); ++i) {
                    if (i == 0) {
                        print_node += "{";
                    }

                    if (printed.find(op->get_input_source_output(i).get_node_shared_ptr()) != printed.end()) {
                        print_node += op->get_input_source_output(i).get_node_shared_ptr()->get_name() + "_pattern";
                    } else {
                        print_node += "ov::pass::pattern::any_input()";
                    }

                    if (i == op->get_input_size() - 1) {
                        print_node += "}";
                    } else {
                        print_node += ", ";
                    }
                }
                print_node += ");";
                os << print_node << std::endl;
                printed.insert(op);
            }
        }
    }
}

// TODO: if empty start_nodes & end_nodes dump the entire graph
static void dump_partially(std::ostream& os,
                           const std::shared_ptr<ov::Model>& model,
                           const std::vector<std::string>& start_nodes,
                           const std::vector<std::string>& end_nodes) {
    std::unordered_set<NodePtr> path_nodes;

    // Find nodes that we would include in our paths and write them to path_nodes
    for (auto& op : model->get_ordered_ops()) {
        for (auto& start_node_name : start_nodes) {
            if (op->get_name() == start_node_name) { // found a node with a given name, we can start processing it
                recurse_down_dfs(op, path_nodes, end_nodes);
            }
        }
    }

    std::unordered_set<NodePtr> printed;

    for (auto& op : model->get_ordered_ops()) {
        if (path_nodes.find(op) != path_nodes.end()) {
            if (auto multi_subgraph = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
                for (size_t i = 0; i < multi_subgraph->get_internal_subgraphs_size(); ++i) {
                    os << "// MultiSubGraphOp " << multi_subgraph->get_name() << "[" << i << "]" << std::endl; // TODO: think if the comment is required here
                    dump_partially(os, multi_subgraph->get_function(i), {}, {});
                }
            } else {
                std::vector<std::string> node_inputs_names;

                // First, process the nodes's inputs
                for (auto& input : op->input_values()) {
                    std::string input_name = input.get_node_shared_ptr()->get_name();
                    if (printed.find(input.get_node_shared_ptr()) == printed.end()) { // it hasn't been printed yet
                        if (auto const_op = std::dynamic_pointer_cast<ov::op::v0::Constant>(input.get_node_shared_ptr())) { // if it's a Constant, we'll just reuse it without faking (same for Parameter)
                            auto print_input_node = "auto " + input_name + " = ov::gen_pattern::makeConst(" + to_code(input.get_element_type()) + ", ov::Shape(" + to_code(input.get_shape()) + "), " + to_code(const_op, true) + ");";
                            os << print_input_node << std::endl;
                        } else {
                            if (!std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr())) { // if it's not a parameter, add "Fake_" to name
                                input_name.insert(0, "Fake_");
                            }
                            auto print_input_node = "auto " + input_name + " = ov::gen_pattern::makeOP<opset1::Parameter>({}, {{\"element_type\", " + to_code(input.get_element_type()) + "}, {\"shape\", " + to_code(input.get_partial_shape()) + "}});";
                            os << print_input_node << std::endl;
                        }
                    }
                    node_inputs_names.push_back(input_name);
                }

                // Now, process the node
                if (auto const_op = std::dynamic_pointer_cast<ov::op::v0::Constant>(op)) {
                    auto print_node = "auto " + op->get_name() + " = ov::gen_pattern::makeConst(" + to_code(op->get_element_type()) + ", ov::Shape(" + to_code(op->get_shape()) + "), " + to_code(const_op, true) + ");";
                    os << print_node << std::endl;
                } else {
                    auto type = op->get_type_info().get_version() + "::" + std::string(op->get_type_name());
                    auto print_node = "auto " + op->get_name() + " = ov::gen_pattern::makeOP<" + type + ">({";
                    for (size_t i = 0; i < node_inputs_names.size(); ++i) {
                        print_node += node_inputs_names[i] + (i == node_inputs_names.size() - 1 ? "}, " : ", ");
                    }
                    std::stringstream ss;
                    OstreamAttributeVisitor osvis(ss);
                    op->visit_attributes(osvis);
                    auto str_attr = ss.str();
                    print_node += "{" + str_attr + "});";
                    os << print_node << std::endl;
                }

                printed.insert(op);
            }
        }
    }

    foo();
}

static void dump_cpp_style_old(std::ostream& os, const std::shared_ptr<ov::Model>& model) {
    const ov::Model& f = *model;
    std::string prefix = "";
    std::string tag = "";
    std::string sep = "";
    os << prefix;
    for (auto op : f.get_results()) {
        os << sep << op->get_name();
        sep = ",";
    }
    os << " " << f.get_friendly_name() << "(\n" << prefix;
    for (auto op : f.get_parameters()) {
        os << "    " << tag << op->get_friendly_name() << ",\n" << prefix;
    }
    os << ") {\n";

    // collect all scalar & short 1D vectors for literal-style display
    std::map<std::shared_ptr<ov::Node>, std::string> literal_consts;
    for (auto op : f.get_ordered_ops()) {
        if (auto constop = std::dynamic_pointer_cast<op::v0::Constant>(op)) {
            // only i32/f32 type const literal can be parsed by C++ compiler
            if (constop->get_output_element_type(0) != ov::element::i32 &&
                constop->get_output_element_type(0) != ov::element::i64 &&
                constop->get_output_element_type(0) != ov::element::f32)
                continue;
            auto shape = constop->get_shape();
            if (shape.size() > 1)
                continue;
            if (shape_size(constop->get_shape()) > 64)
                continue;
            literal_consts[op] = to_code(constop);
        }
    }

    auto get_output_values_info = [](std::shared_ptr<ov::Node>& op) {
        std::stringstream ss;
        const char* sep = "";
        for (size_t i = 0; i < op->get_output_size(); i++) {
            ss << sep << op->get_output_element_type(i) << op->get_output_partial_shape(i);
            sep = " ";
        }
        return ss.str();
    };

    // change name convension
    std::map<ov::Node*, std::string> opname;
    std::map<std::string, int> opname_count;
    for (auto op : f.get_ordered_ops()) {
        auto name = op->get_friendly_name();
        std::replace(name.begin(), name.end(), '\\', '_');
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '.', '_');
        std::replace(name.begin(), name.end(), '[', '_');
        std::replace(name.begin(), name.end(), ']', '_');
        std::replace(name.begin(), name.end(), '-', 'n');
        if (name[0] >= '0' && name[0] <= '9') {
            const auto& type_info = op->get_type_info();
            name.insert(0, type_info.name);
        }
        int idx = 0;
        if (opname_count.count(name)) {
            idx = opname_count[name] + 1;
        }
        opname_count[name] = idx;

        if (idx)
            name += std::to_string(idx);

        opname[op.get()] = name;
    }

    for (auto op : f.get_ordered_ops()) {
        if (literal_consts.count(op))
            continue;

        const auto& type_info = op->get_type_info();
        auto version_info = std::string(type_info.get_version());
        auto type = version_info + "::" + type_info.name;
        auto& rt_info = op->get_rt_info();
        if (rt_info.count("opset") && rt_info["opset"] == "type_relaxed_opset") {
            type = std::string("ov::op::TypeRelaxed<") + type + ">";
        }
        auto name = opname[op.get()];
        os << prefix << "    ";

        if (auto constop = std::dynamic_pointer_cast<op::v0::Constant>(op)) {
            os << "auto " << name << " = makeConst(" << to_code(op->get_output_element_type(0)) << ", "
               << to_code(op->get_output_shape(0)) << ", " << to_code(constop, true) << ");" << std::endl;
        } else {
            os << "auto " << name << " = makeOP<" << type << ">({";
            // input args
            sep = "";
            for (size_t i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << sep << tag << opname[iop.get()] << "->output(" << out_port << ")";
                } else {
                    if (literal_consts.count(iop))
                        os << sep << tag << literal_consts[iop];
                    else
                        os << sep << tag << opname[iop.get()];
                }
                sep = ", ";
            }
            os << "}";

            // attributes as AnyMap
            std::stringstream ss2;
            OstreamAttributeVisitor osvis(ss2);
            op->visit_attributes(osvis);
            auto str_attr = ss2.str();
            if (str_attr.size())
                os << ", {" << str_attr << "}";
            os << ");   //  tensor_array<" << get_output_values_info(op) << "> " << op->get_friendly_name();

            os << "(";
            sep = "";
            for (size_t i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                os << sep << tag << iop->get_friendly_name();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << "[" << out_port << "]";
                }
                sep = ", ";
            }
            os << ")" << std::endl;
        }

        // recursively output subgraphs
        if (auto msubgraph = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
            auto cnt = msubgraph->get_internal_subgraphs_size();
            for (size_t i = 0; i < cnt; i++) {
                os << "    MultiSubGraphOp " << tag << msubgraph->get_friendly_name() << "[" << i << "]" << std::endl;
                dump_cpp_style_old(os, msubgraph->get_function(i));
            }
        }
    }
    os << prefix << "}\n";
}

static void dump_cpp_style(std::ostream& os, const std::shared_ptr<ov::Model>& model, std::vector<std::string> start_nodes, std::vector<std::string> end_nodes, bool use_wrap_type) {
    if (use_wrap_type) {
        dump_wrap_type(os, model, start_nodes, end_nodes);
    } else {
        dump_partially(os, model, start_nodes, end_nodes);
    }
}


}  // namespace detail

class OPENVINO_API PrintModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::pass::PrintModel");

    PrintModel(const std::string& file_name,
               const std::initializer_list<std::string>& start_nodes = {},
               const std::initializer_list<std::string>& end_nodes = {},
               bool use_wrap_type = false)
    : m_start_nodes(start_nodes),
      m_end_nodes(end_nodes),
      m_use_wrap_type(use_wrap_type) {
        static int dump_index = 0;
        m_file_name = std::string("modelprint_") + std::to_string(dump_index) + "_" + file_name;
        dump_index++;
    }
    ~PrintModel() {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        if (m_file_name.empty())
            return false;

        for (auto& node : model->get_ordered_ops()) {
            ov::op::util::process_subgraph(*this, node);
        }

        std::ofstream ofs(m_file_name);
        if (!ofs) {
            return false;
        }

        // if there's no start or end nodes, start dumping from
        // inputs and end with results of the model
        if (m_start_nodes.empty()) {
            for (auto& param: model->get_parameters()) {
                m_start_nodes.push_back(param->get_name());
            }
        }

        if (m_end_nodes.empty()) {
            for (auto& result: model->get_results()) {
                m_end_nodes.push_back(result->get_name());
            }
        }

        detail::dump_cpp_style(ofs, model, m_start_nodes, m_end_nodes, m_use_wrap_type);
        ofs.close();

        return true;
    }

protected:
    std::string m_file_name;
    std::vector<std::string> m_start_nodes;
    std::vector<std::string> m_end_nodes;
    bool m_use_wrap_type;
};
}  // namespace pass
}  // namespace ov
