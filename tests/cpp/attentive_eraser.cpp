// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/attentive_eraser_utils.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "image_generation/attentive_eraser_mask_processor.hpp"
#include "image_generation/stable_diffusion_pipeline.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

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
    using StableDiffusionPipeline::extract_attentive_eraser_aas_layers;
    using StableDiffusionPipeline::process_attentive_mask;

    bool uses_ddim_scheduler() const {
        return std::dynamic_pointer_cast<ov::genai::DDIMScheduler>(m_scheduler) != nullptr;
    }
};

using AdditionalUNetInputs = std::unordered_map<std::string, ov::Tensor>;
static_assert(std::is_same_v<
              decltype(std::declval<ov::genai::UNet2DConditionModel&>().infer(
                  std::declval<ov::Tensor>(),
                  std::declval<ov::Tensor>(),
                  std::declval<const AdditionalUNetInputs&>())),
              ov::Tensor>);

TEST(AttentiveEraserTensorTest, ConvertsRgbMaskToGrayBeforeBinarizing) {
    const std::array<uint8_t, 12> pixels{255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};
    ov::Tensor mask(ov::element::u8, {1, 2, 2, 3}, const_cast<uint8_t*>(pixels.data()));

    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 1, 0.1f, false);
    ov::Tensor processed = processor.execute(mask);

    ASSERT_EQ(processed.get_shape(), ov::Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(processed.data<const float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(processed.data<const float>()[1], 0.0f);
}

TEST(AttentiveEraserTensorTest, ProducesFullResolutionAndPooledMasksInOneInference) {
    const std::array<uint8_t, 16> pixels{
        0, 255, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 255, 0,
    };
    ov::Tensor mask(ov::element::u8, {1, 4, 4, 1}, const_cast<uint8_t*>(pixels.data()));
    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 1, 0.1f, true, 2);

    const auto [full_resolution, pooled] = processor.execute_with_pooling(mask);

    EXPECT_EQ(full_resolution.get_shape(), ov::Shape({1, 1, 4, 4}));
    ASSERT_EQ(pooled.get_shape(), ov::Shape({1, 1, 2, 2}));
    const std::array<float, 4> expected{1.0f, 0.0f, 0.0f, 1.0f};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(pooled.data<const float>()[index], expected[index]);
    }
}

TEST(AttentiveEraserTensorTest, RejectsMaskDimensionsNotDivisibleByPoolingFactor) {
    std::array<uint8_t, 15> pixels{};
    ov::Tensor mask(ov::element::u8, {1, 3, 5, 1}, pixels.data());
    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 1, 0.1f, true, 2);

    EXPECT_THROW(processor.execute_with_pooling(mask), ov::Exception);
}

TEST(AttentiveEraserTensorTest, OpenVinoMaskProcessorMatchesReference) {
    std::array<uint8_t, 9 * 9 * 3> pixels{};
    for (size_t y = 2; y < 6; ++y) {
        for (size_t x = 3; x < 7; ++x) {
            const size_t offset = (y * 9 + x) * 3;
            pixels[offset] = 255;
            pixels[offset + 1] = 255;
            pixels[offset + 2] = 255;
        }
    }
    const std::array<float, 9 * 9> torchvision_expected{
        0, 0, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    ov::Tensor mask(ov::element::u8, {1, 9, 9, 3}, pixels.data());
    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 7, 0.1f, false);
    ov::Tensor processed = processor.execute(mask);

    ASSERT_EQ(processed.get_shape(), ov::Shape({1, 1, 9, 9}));
    for (size_t index = 0; index < processed.get_size(); ++index) {
        EXPECT_FLOAT_EQ(processed.data<const float>()[index], torchvision_expected[index]);
    }
}

TEST(AttentiveEraserTensorTest, OpenVinoMaskProcessorMatchesReferenceForSdxlKernelAndGrayMask) {
    std::array<uint8_t, 79 * 81> pixels{};
    for (size_t y = 25; y < 54; ++y) {
        for (size_t x = 26; x < 55; ++x) {
            pixels[y * 81 + x] = 255;
        }
    }
    ov::Tensor mask(ov::element::u8, {1, 79, 81, 1}, pixels.data());
    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 77, 0.1f, true);
    ov::Tensor processed = processor.execute(mask);

    ASSERT_EQ(processed.get_shape(), ov::Shape({1, 1, 79, 81}));
    for (size_t index = 0; index < processed.get_size(); ++index) {
        EXPECT_TRUE(processed.data<const float>()[index] == 0.0f ||
                    processed.data<const float>()[index] == 1.0f);
    }
}

TEST(AttentiveEraserTensorTest, OpenVinoMaskProcessorRejectsInvalidReflectionPadding) {
    std::array<uint8_t, 3 * 5> pixels{};
    ov::Tensor mask(ov::element::u8, {1, 3, 5, 1}, pixels.data());

    ov::genai::AttentiveEraserMaskProcessor processor("CPU", 7, 0.1f, true);

    EXPECT_THROW(processor.execute(mask), ov::Exception);
}

TEST(AttentiveEraserTensorTest, PipelineRebuildsMaskProcessorWhenKernelChanges) {
    std::array<uint8_t, 7 * 7> pixels{};
    pixels[3 * 7 + 3] = 255;
    ov::Tensor mask(ov::element::u8, {1, 7, 7, 1}, pixels.data());
    AttentiveEraserPipelineTestAccessor pipeline(true);

    ov::Tensor kernel_one = pipeline.process_attentive_mask(mask, 1, 1).full_resolution;
    ov::Tensor kernel_three = pipeline.process_attentive_mask(mask, 3, 1).full_resolution;
    ov::genai::AttentiveEraserMaskProcessor processor_one("CPU", 1, 0.1f, true);
    ov::genai::AttentiveEraserMaskProcessor processor_three("CPU", 3, 0.1f, true);
    ov::Tensor reference_one = processor_one.execute(mask);
    ov::Tensor reference_three = processor_three.execute(mask);

    for (size_t index = 0; index < mask.get_size(); ++index) {
        EXPECT_FLOAT_EQ(kernel_one.data<const float>()[index], reference_one.data<const float>()[index]);
        EXPECT_FLOAT_EQ(kernel_three.data<const float>()[index], reference_three.data<const float>()[index]);
    }
}

TEST(AttentiveEraserTensorTest, AppliesRemovalGuidance) {
    const std::array<float, 4> noise_values{1.0f, 2.0f, 3.0f, 6.0f};
    ov::Tensor noise_pair(ov::element::f32, {2, 1, 1, 2}, const_cast<float*>(noise_values.data()));
    ov::Tensor guided(ov::element::f32, {1, 1, 1, 2});
    float* guided_data = guided.data<float>();

    AttentiveEraserPipelineTestAccessor::apply_attentive_removal_guidance(noise_pair, 2.0f, guided);

    EXPECT_EQ(guided.get_shape(), ov::Shape({1, 1, 1, 2}));
    EXPECT_EQ(guided.data<float>(), guided_data);
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

TEST(AttentiveEraserPipelineTest, ConsumesInternalLayerOverrideBeforeComponentCompilation) {
    AttentiveEraserPipelineTestAccessor pipeline(true);
    ov::AnyMap properties{{"ATTENTIVE_ERASER_AAS_LAYERS", std::vector<size_t>{40, 41, 42}}};

    const auto layers = pipeline.extract_attentive_eraser_aas_layers(properties);

    EXPECT_EQ(layers, (std::vector<size_t>{40, 41, 42}));
    EXPECT_TRUE(properties.empty());
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
                      "Attentive Eraser mode supports only Stable Diffusion 1.5, 2, and SDXL Base pipelines"),
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
                    "StableDiffusionXLImg2ImgPipeline",
                    "StableDiffusionXLRefinerPipeline",
                    "StableDiffusionXLTurboPipeline",
                    "StableDiffusion3Pipeline",
                    "FluxPipeline",
                    "FluxFillPipeline"));

TEST(AttentiveEraserPipelineTest, AcceptsSdxlBaseAtPublicDispatchGate) {
    const auto root_dir = std::filesystem::temp_directory_path() / "attentive_eraser_sdxl_base";
    std::filesystem::create_directories(root_dir);
    {
        std::ofstream model_index(root_dir / "model_index.json");
        model_index << R"({"_class_name": "StableDiffusionXLPipeline", "_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"})";
    }

    try {
        ov::genai::InpaintingPipeline pipeline(
            root_dir,
            "CPU",
            ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));
        FAIL() << "Expected incomplete synthetic SDXL directory to fail after dispatch";
    } catch (const ov::Exception& error) {
        EXPECT_EQ(std::string(error.what()).find("Attentive Eraser mode supports only"), std::string::npos);
    }

    std::filesystem::remove_all(root_dir);
}

TEST(AttentiveEraserPipelineTest, RejectsCustomSdxlModelIdentity) {
    const auto root_dir = std::filesystem::temp_directory_path() / "attentive_eraser_custom_sdxl";
    std::filesystem::create_directories(root_dir);
    {
        std::ofstream model_index(root_dir / "model_index.json");
        model_index << R"({"_class_name": "StableDiffusionXLPipeline", "_name_or_path": "example/custom-sdxl"})";
    }

    try {
        ov::genai::InpaintingPipeline pipeline(
            root_dir,
            "CPU",
            ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));
        FAIL() << "Expected Attentive Eraser to reject a custom SDXL model";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("supports only SDXL Base 1.0"), std::string::npos);
    }

    std::filesystem::remove_all(root_dir);
}

TEST(AttentiveEraserConfigTest, UsesFullDenoisingStrengthForEveryModelFamily) {
    ov::genai::ImageGenerationConfig config;
    config.strength = 0.9999f;
    config.height = 512;
    config.width = 512;

    ov::genai::apply_attentive_eraser_defaults(config);

    EXPECT_FLOAT_EQ(config.strength, 1.0f);
    EXPECT_FLOAT_EQ(config.guidance_scale, 1.0f);
    EXPECT_EQ(config.num_images_per_prompt, 1);
    EXPECT_EQ(config.height, 512);
    EXPECT_EQ(config.width, 512);
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

TEST(AttentiveEraserConfigTest, ProvidesRuntimeAasDefaults) {
    ov::genai::AttentiveEraserConfig config;

    EXPECT_EQ(config.start_step, 0);
    EXPECT_FLOAT_EQ(config.ss_scale, 0.3f);
}

TEST(AttentiveEraserConfigTest, ValidatesRuntimeAasControls) {
    ov::genai::AttentiveEraserConfig config;
    EXPECT_NO_THROW(config.validate());

    config.ss_scale = 0.0f;
    EXPECT_THROW(config.validate(), ov::Exception);

    config.ss_scale = 1.01f;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(AttentiveEraserConfigTest, UsesConfiguredAasStepBoundaries) {
    EXPECT_FALSE(ov::genai::is_attentive_eraser_aas_active(3, 4, 0.8f, 50));
    EXPECT_TRUE(ov::genai::is_attentive_eraser_aas_active(4, 4, 0.8f, 50));
    EXPECT_TRUE(ov::genai::is_attentive_eraser_aas_active(39, 4, 0.8f, 50));
    EXPECT_FALSE(ov::genai::is_attentive_eraser_aas_active(40, 4, 0.8f, 50));
    EXPECT_TRUE(ov::genai::is_attentive_eraser_ss_active(9, 9));
    EXPECT_FALSE(ov::genai::is_attentive_eraser_ss_active(10, 9));
}

TEST(AttentiveEraserModelTest, GeneratesSequentialDynamicSizesWithOnePipeline) {
    const char* model_path = std::getenv("ATTENTIVE_ERASER_MODEL_PATH");
    if (!model_path) {
        GTEST_SKIP() << "ATTENTIVE_ERASER_MODEL_PATH is not set";
    }
    const char* requested_device = std::getenv("ATTENTIVE_ERASER_DEVICE");
    const std::string device = requested_device ? requested_device : "CPU";
    ov::genai::InpaintingPipeline pipeline(
        model_path,
        device,
        ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));

    ov::Tensor image(ov::element::u8, {1, 512, 512, 3});
    ov::Tensor mask(ov::element::u8, {1, 512, 512, 3});
    std::fill_n(image.data<uint8_t>(), image.get_size(), uint8_t{127});
    std::fill_n(mask.data<uint8_t>(), mask.get_size(), uint8_t{0});

    const std::array<std::pair<int64_t, int64_t>, 4> sizes{{
        {512, 512},
        {768, 512},
        {512, 768},
        {512, 512},
    }};
    std::vector<uint8_t> first_output;
    for (const auto& [height, width] : sizes) {
        auto config = pipeline.get_generation_config();
        config.height = height;
        config.width = width;
        config.strength = 0.8f;
        config.num_inference_steps = 2;
        config.rng_seed = 123;
        pipeline.set_generation_config(config);

        ov::Tensor output = pipeline.generate("", image, mask);
        ASSERT_EQ(output.get_shape(), ov::Shape({1, static_cast<size_t>(height), static_cast<size_t>(width), 3}));
        ASSERT_EQ(output.get_element_type(), ov::element::u8);
        if (first_output.empty()) {
            first_output.assign(output.data<const uint8_t>(), output.data<const uint8_t>() + output.get_size());
        } else if (height == 512 && width == 512) {
            int max_difference = 0;
            for (size_t index = 0; index < output.get_size(); ++index) {
                max_difference = std::max(
                    max_difference,
                    std::abs(static_cast<int>(output.data<const uint8_t>()[index]) - first_output[index]));
            }
            EXPECT_LE(max_difference, device == "CPU" ? 0 : 2);
        }
    }
}

}  // namespace
