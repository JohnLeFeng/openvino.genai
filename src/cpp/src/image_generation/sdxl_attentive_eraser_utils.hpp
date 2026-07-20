// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {
namespace attentive_eraser {

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
