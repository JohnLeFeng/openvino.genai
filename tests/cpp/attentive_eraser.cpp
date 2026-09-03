// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/attentive_eraser_utils.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "image_generation/stable_diffusion_pipeline.hpp"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>

#include "image_generation/schedulers/ddim.hpp"

namespace {

class AttentiveEraserPipelineTestAccessor : public ov::genai::StableDiffusionPipeline {
public:
    explicit AttentiveEraserPipelineTestAccessor(bool use_attentive_eraser) :
        StableDiffusionPipeline(ov::genai::PipelineType::INPAINTING) {
        m_use_attentive_eraser = use_attentive_eraser;
    }

    using StableDiffusionPipeline::apply_attentive_removal_guidance;
    using StableDiffusionPipeline::blend_attentive_latents;

    bool uses_ddim_scheduler() const {
        return std::dynamic_pointer_cast<ov::genai::DDIMScheduler>(m_scheduler) != nullptr;
    }
};

TEST(AttentiveEraserTensorTest, ConvertsRgbMaskToGrayBeforeBinarizing) {
    const std::array<uint8_t, 12> pixels{255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};
    ov::Tensor mask(ov::element::u8, {1, 2, 2, 3}, const_cast<uint8_t*>(pixels.data()));

    ov::Tensor processed = ov::genai::preprocess_attentive_mask(mask, 1, 0.1f);

    ASSERT_EQ(processed.get_shape(), ov::Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(processed.data<const float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[1], 0.0f);
}

TEST(AttentiveEraserTensorTest, AppliesRemovalGuidance) {
    const std::array<float, 4> noise_values{1.0f, 2.0f, 3.0f, 6.0f};
    ov::Tensor noise_pair(ov::element::f32, {2, 1, 1, 2}, const_cast<float*>(noise_values.data()));

    ov::Tensor guided = AttentiveEraserPipelineTestAccessor::apply_attentive_removal_guidance(noise_pair, 2.0f);

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

    AttentiveEraserPipelineTestAccessor::blend_attentive_latents(initial, mask, latents);

    EXPECT_FLOAT_EQ(latents.data<const float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(latents.data<const float>()[1], 20.0f);
}

TEST(AttentiveEraserSchedulerTest, RejectsNonDdimOverrideAndKeepsCurrentScheduler) {
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

    AttentiveEraserPipelineTestAccessor pipeline(true);
    pipeline.set_scheduler(ov::genai::Scheduler::from_config(config_path, ov::genai::Scheduler::Type::DDIM));
    ov::genai::DiffusionPipeline& pipeline_interface = pipeline;

    EXPECT_THROW(pipeline_interface.set_scheduler(ov::genai::Scheduler::from_config(config_path)), ov::Exception);
    EXPECT_TRUE(pipeline.uses_ddim_scheduler());
    std::filesystem::remove(config_path);
}

class UnsupportedAttentiveEraserPipelineTest : public testing::TestWithParam<const char*> {};

TEST_P(UnsupportedAttentiveEraserPipelineTest, RejectsUnsupportedModelFamily) {
    const std::string class_name = GetParam();
    const auto root_dir = std::filesystem::temp_directory_path() / ("attentive_eraser_" + class_name);
    std::filesystem::create_directories(root_dir);
    {
        std::ofstream model_index(root_dir / "model_index.json");
        model_index << R"({"_class_name": ")" << class_name << R"("})";
    }

    try {
        ov::genai::InpaintingPipeline pipeline(
            root_dir,
            "CPU",
            ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));
        FAIL() << "Expected Attentive Eraser to reject " << class_name;
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find(
                      "Attentive Eraser mode requires a 4-channel SD1.5, SD2, or SDXL pipeline"),
                  std::string::npos);
    }

    std::filesystem::remove_all(root_dir);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedModelFamilies,
    UnsupportedAttentiveEraserPipelineTest,
    testing::Values("LatentConsistencyModelPipeline",
                    "StableDiffusionInpaintPipeline",
                    "StableDiffusionXLInpaintPipeline",
                    "StableDiffusion3Pipeline",
                    "FluxPipeline",
                    "FluxFillPipeline"));

TEST(AttentiveEraserConfigTest, UsesFullDenoisingStrengthForEveryModelFamily) {
    ov::genai::ImageGenerationConfig config;
    config.strength = 0.9999f;

    ov::genai::apply_attentive_eraser_defaults(config);

    EXPECT_FLOAT_EQ(config.strength, 1.0f);
    EXPECT_FLOAT_EQ(config.guidance_scale, 1.0f);
    EXPECT_EQ(config.num_images_per_prompt, 1);
    EXPECT_TRUE(config.attentive_eraser.has_value());
}

TEST(AttentiveEraserConfigTest, ValidatesMaskBlurKernelOverride) {
    ov::genai::AttentiveEraserConfig config;
    EXPECT_EQ(config.mask_blur_kernel, 0);
    EXPECT_NO_THROW(config.validate());

    config.mask_blur_kernel = 9;
    EXPECT_NO_THROW(config.validate());

    config.mask_blur_kernel = 8;
    EXPECT_THROW(config.validate(), ov::Exception);
}

}  // namespace
