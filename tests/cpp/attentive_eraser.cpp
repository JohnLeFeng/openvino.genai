// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/attentive_eraser_utils.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "image_generation/stable_diffusion_pipeline.hpp"

#include <gtest/gtest.h>

#include <array>

namespace {

class AttentiveEraserPipelineTestAccessor : public ov::genai::StableDiffusionPipeline {
public:
    using StableDiffusionPipeline::apply_attentive_removal_guidance;
    using StableDiffusionPipeline::blend_attentive_latents;
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
