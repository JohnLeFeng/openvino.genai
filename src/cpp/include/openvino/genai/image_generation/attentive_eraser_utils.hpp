// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

OPENVINO_GENAI_EXPORTS ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor);

OPENVINO_GENAI_EXPORTS ov::Tensor preprocess_attentive_mask(const ov::Tensor& mask,
                                                            size_t kernel_size,
                                                            float threshold);

OPENVINO_GENAI_EXPORTS bool is_attentive_eraser_aas_active(size_t inference_step,
                                                           size_t start_step,
                                                           float strength,
                                                           size_t num_inference_steps);

OPENVINO_GENAI_EXPORTS bool is_attentive_eraser_ss_active(size_t inference_step, size_t ss_steps);

}  // namespace genai
}  // namespace ov