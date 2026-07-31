// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/attentive_eraser.hpp"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include "image_generation/schedulers/ddim.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

std::shared_ptr<ov::Model> make_unet(const std::vector<std::string>& input_names,
                                     ov::element::Type element_type = ov::element::f32) {
    ov::ParameterVector parameters;
    for (const auto& input_name : input_names) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape{1});
        parameter->set_friendly_name(input_name);
        parameter->output(0).get_tensor().set_names({input_name});
        parameters.push_back(parameter);
    }

    auto sum = std::make_shared<ov::op::v1::Add>(parameters[0], parameters[1]);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sum)}, parameters);
}

std::shared_ptr<ov::Model> make_dynamic_sd_unet() {
    ov::ParameterVector parameters;
    const std::vector<std::pair<std::string, ov::PartialShape>> inputs{{"sample", {-1, 4, -1, -1}},
                                                                       {"timestep", {}},
                                                                       {"encoder_hidden_states", {-1, -1, -1}},
                                                                       {"mask", {-1, -1, -1, -1}},
                                                                       {"cur_step", {}},
                                                                       {"ss_steps", {}}};
    for (const auto& input : inputs) {
        const bool is_integer = input.first == "timestep" || input.first == "cur_step" || input.first == "ss_steps";
        auto parameter =
            std::make_shared<ov::op::v0::Parameter>(is_integer ? ov::element::i64 : ov::element::f16, input.second);
        parameter->output(0).get_tensor().set_names({input.first});
        parameters.push_back(parameter);
    }
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(parameters[0])},
                                       parameters);
}

const std::vector<std::string> sd_attentive_inputs{"sample",
                                                   "timestep",
                                                   "encoder_hidden_states",
                                                   "mask",
                                                   "cur_step",
                                                   "ss_steps"};
const std::vector<std::string> sdxl_attentive_inputs{"sample",
                                                     "timestep",
                                                     "encoder_hidden_states",
                                                     "text_embeds",
                                                     "time_ids",
                                                     "mask",
                                                     "cur_step",
                                                     "ss_steps"};

TEST(AttentiveEraserContractTest, AcceptsMatchingAttentiveGraphs) {
    EXPECT_NO_THROW(
        ov::genai::validate_attentive_eraser_unet_inputs(make_unet(sd_attentive_inputs),
                                                         ov::genai::AttentiveEraserModelFamily::STABLE_DIFFUSION,
                                                         true));
    EXPECT_NO_THROW(
        ov::genai::validate_attentive_eraser_unet_inputs(make_unet(sdxl_attentive_inputs),
                                                         ov::genai::AttentiveEraserModelFamily::STABLE_DIFFUSION_XL,
                                                         true));
}

TEST(AttentiveEraserContractTest, RejectsModeAndGraphMismatch) {
    const std::vector<std::string> standard_inputs{"sample", "timestep", "encoder_hidden_states"};

    EXPECT_THROW(
        ov::genai::validate_attentive_eraser_unet_inputs(make_unet(standard_inputs),
                                                         ov::genai::AttentiveEraserModelFamily::STABLE_DIFFUSION,
                                                         true),
        ov::Exception);
    EXPECT_THROW(
        ov::genai::validate_attentive_eraser_unet_inputs(make_unet(sd_attentive_inputs),
                                                         ov::genai::AttentiveEraserModelFamily::STABLE_DIFFUSION,
                                                         false),
        ov::Exception);
}

TEST(AttentiveEraserContractTest, RejectsWrongModelFamilyContract) {
    EXPECT_THROW(
        ov::genai::validate_attentive_eraser_unet_inputs(make_unet(sd_attentive_inputs),
                                                         ov::genai::AttentiveEraserModelFamily::STABLE_DIFFUSION_XL,
                                                         true),
        ov::Exception);
}

TEST(AttentiveEraserContractTest, ExposesFloatingPointInputsAndOutputAsF32) {
    auto model = make_unet(sd_attentive_inputs, ov::element::f16);

    model = ov::genai::prepare_attentive_eraser_unet_model(model);

    EXPECT_EQ(model->input("sample").get_element_type(), ov::element::f32);
    EXPECT_EQ(model->input("encoder_hidden_states").get_element_type(), ov::element::f32);
    EXPECT_EQ(model->input("mask").get_element_type(), ov::element::f32);
    EXPECT_EQ(model->output().get_element_type(), ov::element::f32);
}

TEST(AttentiveEraserContractTest, ReshapesSdGraphToFixedContract) {
    auto model = make_dynamic_sd_unet();

    ov::genai::reshape_attentive_eraser_unet_model(model, 64, 8, 768);

    EXPECT_EQ(model->input("sample").get_partial_shape(), ov::PartialShape({2, 4, 64, 64}));
    EXPECT_EQ(model->input("encoder_hidden_states").get_partial_shape(), ov::PartialShape({2, 77, 768}));
    EXPECT_EQ(model->input("mask").get_partial_shape(), ov::PartialShape({1, 1, 512, 512}));
}

TEST(AttentiveEraserTensorTest, PreprocessesImageFromNhwcU8ToNchwF32) {
    const std::array<uint8_t, 6> pixels{0, 127, 255, 255, 127, 0};
    ov::Tensor image(ov::element::u8, {1, 1, 2, 3}, const_cast<uint8_t*>(pixels.data()));

    ov::Tensor processed = ov::genai::attentive_eraser::preprocess_image(image);

    EXPECT_EQ(processed.get_shape(), ov::Shape({1, 3, 1, 2}));
    EXPECT_FLOAT_EQ(processed.data<const float>()[0], -1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[1], 1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[4], 1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[5], -1.0f);
}

TEST(AttentiveEraserTensorTest, ConvertsRgbMaskToGrayBeforeBinarizing) {
    const std::array<uint8_t, 12> pixels{255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};
    ov::Tensor mask(ov::element::u8, {1, 2, 2, 3}, const_cast<uint8_t*>(pixels.data()));

    ov::Tensor processed = ov::genai::attentive_eraser::preprocess_mask(mask, 1, 0.1f);

    ASSERT_EQ(processed.get_shape(), ov::Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(processed.data<const float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[1], 0.0f);
}

TEST(AttentiveEraserTensorTest, ValidatesFixedInputContract) {
    ov::Tensor image(ov::element::u8, {1, 8, 8, 3});
    ov::Tensor mask(ov::element::u8, {1, 8, 8, 1});
    EXPECT_NO_THROW(ov::genai::attentive_eraser::validate_input_tensor(image, "Initial image", false, 8));
    EXPECT_NO_THROW(ov::genai::attentive_eraser::validate_input_tensor(mask, "Mask image", true, 8));

    ov::Tensor wrong_size(ov::element::u8, {1, 4, 8, 3});
    EXPECT_THROW(ov::genai::attentive_eraser::validate_input_tensor(wrong_size, "Initial image", false, 8),
                 ov::Exception);
}

TEST(AttentiveEraserTensorTest, AppliesRemovalGuidance) {
    const std::array<float, 4> noise_values{1.0f, 2.0f, 3.0f, 6.0f};
    ov::Tensor noise_pair(ov::element::f32, {2, 1, 1, 2}, const_cast<float*>(noise_values.data()));

    ov::Tensor guided = ov::genai::attentive_eraser::removal_guidance(noise_pair, 2.0f);

    EXPECT_EQ(guided.get_shape(), ov::Shape({1, 1, 1, 2}));
    EXPECT_FLOAT_EQ(guided.data<const float>()[0], 5.0f);
    EXPECT_FLOAT_EQ(guided.data<const float>()[1], 10.0f);
}

TEST(AttentiveEraserTensorTest, BlendsLatentsUsingMask) {
    const std::array<float, 2> initial_values{2.0f, 4.0f};
    const std::array<float, 2> mask_values{0.0f, 1.0f};
    std::array<float, 2> latent_values{10.0f, 20.0f};
    ov::Tensor initial(ov::element::f32, {1, 1, 1, 2}, const_cast<float*>(initial_values.data()));
    ov::Tensor mask(ov::element::f32, {1, 1, 1, 2}, const_cast<float*>(mask_values.data()));
    ov::Tensor latents(ov::element::f32, {1, 1, 1, 2}, latent_values.data());

    ov::genai::attentive_eraser::blend_latents(initial, mask, latents);

    EXPECT_FLOAT_EQ(latents.data<const float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(latents.data<const float>()[1], 20.0f);
}

TEST(AttentiveEraserSchedulerTest, ForcesDdimForAttentiveMode) {
    const auto config_path = std::filesystem::temp_directory_path() / "attentive_eraser_scheduler_config.json";
    {
        std::ofstream config(config_path);
        config << R"({
			"_class_name": "PNDMScheduler",
			"beta_start": 0.00085,
			"beta_end": 0.012,
			"beta_schedule": "scaled_linear",
			"clip_sample": false,
			"num_train_timesteps": 1000,
			"prediction_type": "epsilon",
			"set_alpha_to_one": false,
			"steps_offset": 1,
			"timestep_spacing": "leading"
		})";
    }

    auto scheduler = ov::genai::create_attentive_eraser_scheduler(config_path, true);

    EXPECT_NE(std::dynamic_pointer_cast<ov::genai::DDIMScheduler>(scheduler), nullptr);
    std::filesystem::remove(config_path);
}

}  // namespace
