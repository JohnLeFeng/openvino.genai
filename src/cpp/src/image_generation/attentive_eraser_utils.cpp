// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/attentive_eraser_utils.hpp"

namespace ov {
namespace genai {

bool is_attentive_eraser_aas_active(size_t inference_step,
                                    size_t start_step,
                                    float strength,
                                    size_t num_inference_steps) {
    const size_t end_step = static_cast<size_t>(strength * num_inference_steps);
    return inference_step >= start_step && inference_step < end_step;
}

bool is_attentive_eraser_ss_active(size_t inference_step, size_t ss_steps) {
    return inference_step <= ss_steps;
}

}  // namespace genai
}  // namespace ov
