// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/models/attentive_eraser_aas.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov {
namespace genai {
namespace {

constexpr size_t SD15_SELF_ATTENTION_COUNT = 16;
constexpr size_t SDXL_SELF_ATTENTION_COUNT = 70;

bool is_self_attention(const std::shared_ptr<ov::Node>& node) {
    auto attention = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
    if (!attention || attention->get_friendly_name().find(".attn1/") == std::string::npos) {
        return false;
    }

    const auto query_shape = attention->get_input_partial_shape(0);
    const auto key_shape = attention->get_input_partial_shape(1);
    const auto value_shape = attention->get_input_partial_shape(2);
    return query_shape.rank().compatible(4) && key_shape.rank().compatible(4) && value_shape.rank().compatible(4) &&
           query_shape[2].compatible(key_shape[2]) && key_shape[2].compatible(value_shape[2]);
}

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const std::string& name, const ov::PartialShape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).get_tensor().set_names({name});
    return parameter;
}

ov::Output<ov::Node> scalar_to(const ov::Output<ov::Node>& value, const ov::element::Type& type) {
    return std::make_shared<ov::opset13::Convert>(value, type);
}

float lowest_finite_value(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return -65504.0f;
    }
    if (type == ov::element::bf16) {
        return -3.38953139e38f;
    }
    return std::numeric_limits<float>::lowest();
}

ov::Output<ov::Node> attention(const ov::Output<ov::Node>& query,
                               const ov::Output<ov::Node>& key,
                               const ov::Output<ov::Node>& value,
                               const ov::Output<ov::Node>& penalty,
                               const ov::Output<ov::Node>& logits_scale) {
    using namespace ov::opset13;
    auto key_transposed = std::make_shared<Transpose>(key, Constant::create(ov::element::i32, {4}, {0, 1, 3, 2}));
    auto logits = std::make_shared<MatMul>(query, key_transposed);
    auto scaled_logits = std::make_shared<Multiply>(logits, logits_scale);
    auto probabilities = std::make_shared<Softmax>(std::make_shared<Add>(scaled_logits, penalty), -1);
    return std::make_shared<MatMul>(probabilities, value);
}

}  // namespace

void apply_attentive_eraser_aas(const std::shared_ptr<ov::Model>& model,
                                const std::vector<size_t>& layer_indices) {
    OPENVINO_ASSERT(model, "Attentive Eraser AAS requires a UNet model");

    const auto has_input = [&model](const std::string& name) {
        const auto inputs = model->inputs();
        return std::any_of(inputs.begin(), inputs.end(), [&name](const ov::Output<const ov::Node>& input) {
            return input.get_names().count(name) > 0;
        });
    };
    OPENVINO_ASSERT(!has_input("mask") && !has_input("cur_step") && !has_input("ss_steps"),
                    "Legacy pre-converted Attentive Eraser UNet IR is not supported; use an ordinary SD1.5 UNet");

    std::vector<std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> self_attention_layers;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_self_attention(node)) {
            self_attention_layers.push_back(ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node));
        }
    }

    const size_t self_attention_count = self_attention_layers.size();
    OPENVINO_ASSERT(self_attention_count == SD15_SELF_ATTENTION_COUNT ||
                        self_attention_count == SDXL_SELF_ATTENTION_COUNT,
                    "Attentive Eraser AAS requires exactly 16 SD1.5 or 70 SDXL self-attention layers, but found ",
                    self_attention_layers.size());
    OPENVINO_ASSERT(!layer_indices.empty(), "Attentive Eraser AAS requires at least one selected layer");
    OPENVINO_ASSERT(std::all_of(layer_indices.begin(), layer_indices.end(), [self_attention_count](size_t index) {
                        return index < self_attention_count;
                    }),
                    "Attentive Eraser AAS layer indices must be in [0, ", self_attention_count, ")");

    const size_t mask_size = self_attention_count == SDXL_SELF_ATTENTION_COUNT ? 1024 : 512;

    ov::ParameterVector runtime_parameters{
        make_parameter("aas_mask", {1, 1, mask_size, mask_size}),
        make_parameter("aas_active", {}),
        make_parameter("ss_active", {}),
        make_parameter("ss_scale", {}),
    };
    model->add_parameters(runtime_parameters);

    using namespace ov::opset13;
    const auto& mask = runtime_parameters[0];
    const auto& aas_active = runtime_parameters[1];
    const auto& ss_active = runtime_parameters[2];
    const auto& ss_scale = runtime_parameters[3];
    const std::set<size_t> selected_layers(layer_indices.begin(), layer_indices.end());

    for (size_t index = 0; index < self_attention_layers.size(); ++index) {
        if (selected_layers.count(index) == 0) {
            continue;
        }

        const auto& original = self_attention_layers[index];
        OPENVINO_ASSERT(original->get_input_size() == 3,
                        "Attentive Eraser AAS supports self-attention without an existing mask or scale input");
        const auto query_shape = original->get_input_partial_shape(0);
        OPENVINO_ASSERT(query_shape[3].is_static(),
                        "Attentive Eraser AAS requires a static self-attention head dimension");
        const auto data_type = original->get_input_element_type(0);

        auto batch_axis = Constant::create(ov::element::i32, {}, {0});
        auto query_split = std::make_shared<Split>(original->input_value(0), batch_axis, 2);
        auto key_split = std::make_shared<Split>(original->input_value(1), batch_axis, 2);
        auto value_split = std::make_shared<Split>(original->input_value(2), batch_axis, 2);

        auto query_shape_node = std::make_shared<ShapeOf>(original->input_value(0), ov::element::i32);
        auto token_count = std::make_shared<Gather>(query_shape_node,
                                Constant::create(ov::element::i32, {}, {2}),
                                Constant::create(ov::element::i32, {}, {0}));
        auto side_f32 = std::make_shared<Sqrt>(std::make_shared<Convert>(token_count, ov::element::f32));
        auto side = std::make_shared<Convert>(side_f32, ov::element::i32);
        auto side_vector = std::make_shared<Unsqueeze>(side, Constant::create(ov::element::i32, {1}, {0}));
        auto pooled_mask = std::make_shared<AdaptiveMaxPool>(mask,
                                                            std::make_shared<Concat>(ov::OutputVector{side_vector,
                                                                                                    side_vector},
                                                                                     0),
                                                            ov::element::i32);
        auto key_mask = std::make_shared<Reshape>(pooled_mask->output(0),
                                                                                                    Constant::create(ov::element::i32, {4}, {1, 1, 1, -1}),
                                                  false);
        auto query_mask = std::make_shared<Reshape>(pooled_mask->output(0),
                                                                                                        Constant::create(ov::element::i32, {4}, {1, 1, -1, 1}),
                                                    false);

        auto active = scalar_to(aas_active, data_type);
        auto suppression = std::make_shared<Multiply>(scalar_to(key_mask, data_type), active);
        auto penalty = std::make_shared<Multiply>(suppression,
                                                  Constant::create(data_type, {}, {lowest_finite_value(data_type)}));
        const float base_scale = 1.0f / std::sqrt(static_cast<float>(query_shape[3].get_length()));
        auto base_scale_node = Constant::create(data_type, {}, {base_scale});
        auto ss_gate = std::make_shared<Multiply>(active, scalar_to(ss_active, data_type));
        auto scale_delta = std::make_shared<Multiply>(ss_gate,
                                                      std::make_shared<Subtract>(scalar_to(ss_scale, data_type),
                                                                                 Constant::create(data_type, {}, {1.0f})));
        auto foreground_scale = std::make_shared<Multiply>(base_scale_node,
                                                           std::make_shared<Add>(Constant::create(data_type, {}, {1.0f}),
                                                                                 scale_delta));

        auto source = std::make_shared<ScaledDotProductAttention>(query_split->output(0),
                                                                  key_split->output(0),
                                                                  value_split->output(0),
                                                                  false);
        auto background = attention(query_split->output(1),
                                    key_split->output(1),
                                    value_split->output(1),
                                    penalty,
                                    base_scale_node);
        auto foreground = attention(query_split->output(1),
                                    key_split->output(1),
                                    value_split->output(1),
                                    penalty,
                                    foreground_scale);
        auto mixed_delta = std::make_shared<Multiply>(std::make_shared<Subtract>(foreground, background),
                                                      std::make_shared<Multiply>(ss_gate,
                                                                                 scalar_to(query_mask, data_type)));
        auto target = std::make_shared<Add>(background, mixed_delta);
        auto replacement = std::make_shared<Concat>(ov::OutputVector{source, target}, 0);
        replacement->set_friendly_name(original->get_friendly_name());
        ov::replace_node(original, replacement);
    }
}

}  // namespace genai
}  // namespace ov