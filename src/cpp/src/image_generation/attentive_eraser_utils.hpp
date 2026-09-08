// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor);

ov::Tensor preprocess_attentive_mask(const ov::Tensor& mask,
                                     size_t kernel_size,
                                     float threshold);

bool is_attentive_eraser_aas_active(size_t inference_step,
                                    size_t start_step,
                                    float strength,
                                    size_t num_inference_steps);

bool is_attentive_eraser_ss_active(size_t inference_step, size_t ss_steps);

}  // namespace genai
}  // namespace ov
