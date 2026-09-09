// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "image_generation/models/attentive_eraser_aas.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> make_attention_model(size_t self_attention_count) {
    using namespace ov::opset13;

    ov::ResultVector results;
    auto self_qkv = Constant::create(ov::element::f32, {2, 2, 4, 3}, {0.1f});
    auto cross_kv = Constant::create(ov::element::f32, {2, 2, 77, 3}, {0.1f});

    for (size_t index = 0; index < self_attention_count; ++index) {
        auto self_attention = std::make_shared<ScaledDotProductAttention>(self_qkv, self_qkv, self_qkv, false);
        self_attention->set_friendly_name("block." + std::to_string(index) + ".attn1/ScaledDotProductAttention");
        results.push_back(std::make_shared<Result>(self_attention));
    }

    for (size_t index = 0; index < 16; ++index) {
        auto cross_attention = std::make_shared<ScaledDotProductAttention>(self_qkv, cross_kv, cross_kv, false);
        cross_attention->set_friendly_name("block." + std::to_string(index) + ".attn2/ScaledDotProductAttention");
        results.push_back(std::make_shared<Result>(cross_attention));
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{});
}

std::shared_ptr<ov::Model> make_runtime_attention_model() {
    using namespace ov::opset13;

    auto qkv = std::make_shared<Parameter>(ov::element::f32, ov::Shape{2, 1, 4, 1});
    qkv->set_friendly_name("qkv");
    qkv->output(0).get_tensor().set_names({"qkv"});
    ov::ResultVector results;
    for (size_t index = 0; index < 16; ++index) {
        auto attention = std::make_shared<ScaledDotProductAttention>(qkv, qkv, qkv, false);
        attention->set_friendly_name("block." + std::to_string(index) + ".attn1/ScaledDotProductAttention");
        results.push_back(std::make_shared<Result>(attention));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{qkv});
}

size_t count_named_self_attention_nodes(const std::shared_ptr<ov::Model>& model) {
    const auto ordered_ops = model->get_ordered_ops();
    return std::count_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& node) {
        return ov::is_type<ov::opset13::ScaledDotProductAttention>(node) &&
               node->get_friendly_name().find(".attn1/") != std::string::npos;
    });
}

TEST(AttentiveEraserGraphTransform, MatchesAllSelfAttentionLayersBeforeChangingModel) {
    auto complete_model = make_attention_model(16);
    ov::genai::apply_attentive_eraser_aas(complete_model, {7, 8, 9, 10, 11, 12, 13, 14, 15});

    std::set<std::string> input_names;
    for (const auto& input : complete_model->inputs()) {
        input_names.insert(input.get_any_name());
    }
    EXPECT_EQ(input_names, (std::set<std::string>{"aas_active", "aas_mask", "ss_active", "ss_scale"}));

    auto incomplete_model = make_attention_model(15);
    EXPECT_THROW(ov::genai::apply_attentive_eraser_aas(incomplete_model, {7, 8, 9, 10, 11, 12, 13, 14, 15}),
                 ov::Exception);
    EXPECT_TRUE(incomplete_model->inputs().empty());
}

TEST(AttentiveEraserGraphTransform, TransformsStrictSdxlTopologyFromLayer34) {
    auto complete_model = make_attention_model(70);
    std::vector<size_t> selected_layers(36);
    std::iota(selected_layers.begin(), selected_layers.end(), 34);

    ov::genai::apply_attentive_eraser_aas(complete_model, selected_layers);

    EXPECT_EQ(count_named_self_attention_nodes(complete_model), 34);
    EXPECT_EQ(complete_model->input("aas_mask").get_partial_shape(), ov::PartialShape({1, 1, 1024, 1024}));

    auto incomplete_model = make_attention_model(69);
    EXPECT_THROW(ov::genai::apply_attentive_eraser_aas(incomplete_model, selected_layers), ov::Exception);
    EXPECT_TRUE(incomplete_model->inputs().empty());
}

TEST(AttentiveEraserGraphTransform, SupportsInternalSdxlLayerOverrideFromLayer40) {
    auto model = make_attention_model(70);
    std::vector<size_t> selected_layers(30);
    std::iota(selected_layers.begin(), selected_layers.end(), 40);

    ov::genai::apply_attentive_eraser_aas(model, selected_layers);

    EXPECT_EQ(count_named_self_attention_nodes(model), 40);
    EXPECT_EQ(model->inputs().size(), 4);
}

TEST(AttentiveEraserGraphTransform, RejectsLegacyPreconvertedUnetBeforeChangingModel) {
    auto model = make_attention_model(16);
    auto legacy_mask = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::Shape{1, 1, 512, 512});
    legacy_mask->output(0).get_tensor().set_names({"mask"});
    model->add_parameters({legacy_mask});

    EXPECT_THROW(ov::genai::apply_attentive_eraser_aas(model, {7, 8, 9, 10, 11, 12, 13, 14, 15}), ov::Exception);
    ASSERT_EQ(model->inputs().size(), 1);
    EXPECT_EQ(model->input().get_any_name(), "mask");
}

TEST(AttentiveEraserGraphTransform, AppliesMaskAndSoftmaxScalingToSelectedLayer) {
    auto model = make_runtime_attention_model();
    ov::genai::apply_attentive_eraser_aas(model, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto compiled = ov::genai::utils::singleton_core().compile_model(model, "CPU");
    auto request = compiled.create_infer_request();

    ov::Tensor qkv{ov::element::f32, {2, 1, 4, 1}};
    std::copy_n(std::array<float, 8>{1, 2, 3, 4, 1, 2, 3, 4}.begin(), 8, qkv.data<float>());
    ov::Tensor mask{ov::element::f32, {1, 1, 512, 512}};
    std::fill_n(mask.data<float>(), mask.get_size(), 0.0f);
    std::fill_n(mask.data<float>(), 256 * 512, 1.0f);
    ov::Tensor aas_active{ov::element::f32, {}};
    ov::Tensor ss_active{ov::element::f32, {}};
    ov::Tensor ss_scale{ov::element::f32, {}};
    aas_active.data<float>()[0] = 1.0f;
    ss_active.data<float>()[0] = 1.0f;
    ss_scale.data<float>()[0] = 0.5f;
    request.set_tensor("qkv", qkv);
    request.set_tensor("aas_mask", mask);
    request.set_tensor("aas_active", aas_active);
    request.set_tensor("ss_active", ss_active);
    request.set_tensor("ss_scale", ss_scale);
    request.infer();

    const auto output = request.get_output_tensor(0);
    const auto* values = output.data<const float>();
    EXPECT_NEAR(values[0], 3.49265f, 1e-3f);
    EXPECT_NEAR(values[4], 3.62246f, 1e-3f);
    EXPECT_NEAR(values[6], 3.95257f, 1e-3f);

    aas_active.data<float>()[0] = 0.0f;
    request.infer();
    const auto inactive_output = request.get_output_tensor(0);
    const auto* inactive_values = inactive_output.data<const float>();
    EXPECT_NEAR(inactive_values[4], 3.49265f, 1e-3f);
    EXPECT_NEAR(inactive_values[6], 3.94763f, 1e-3f);
}

}  // namespace