// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

enum class AttentiveEraserModelFamily {
    STABLE_DIFFUSION,
    STABLE_DIFFUSION_XL,
};

void validate_attentive_eraser_unet_inputs(const std::shared_ptr<ov::Model>& model,
                                           AttentiveEraserModelFamily model_family,
                                           bool attentive_eraser_enabled);

std::shared_ptr<ov::Model> prepare_attentive_eraser_unet_model(std::shared_ptr<ov::Model> model);

void reshape_attentive_eraser_unet_model(const std::shared_ptr<ov::Model>& model,
                                         size_t sample_size,
                                         size_t vae_scale_factor,
                                         size_t cross_attention_dim);

std::shared_ptr<Scheduler> create_attentive_eraser_scheduler(
    const std::filesystem::path& scheduler_config_path,
    bool attentive_eraser_enabled);

namespace attentive_eraser {

void validate_input_tensor(const ov::Tensor& tensor,
                           const char* name,
                           bool is_mask,
                           size_t image_size);

ov::Tensor preprocess_image(const ov::Tensor& image);

ov::Tensor preprocess_mask(const ov::Tensor& mask,
                           size_t kernel_size,
                           float threshold);

ov::Tensor gaussian_blur_and_binarize_mask(const ov::Tensor& gray_mask,
                                            size_t kernel_size,
                                            float threshold);

ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor);

ov::Tensor removal_guidance(const ov::Tensor& noise_pair, float scale);

void blend_latents(const ov::Tensor& initial_noised,
                   const ov::Tensor& mask,
                   ov::Tensor latents);

}  // namespace attentive_eraser

}  // namespace genai
}  // namespace ov