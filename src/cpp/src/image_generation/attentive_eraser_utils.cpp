// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/attentive_eraser_utils.hpp"

#include <algorithm>

namespace ov {
namespace genai {

namespace {

void validate_f32_nchw(const ov::Tensor& tensor, const char* name) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32, name, " must have f32 element type");
    OPENVINO_ASSERT(tensor.get_shape().size() == 4, name, " must be a rank-4 NCHW tensor");
}

}  // namespace

ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor) {
    validate_f32_nchw(mask, "Mask");
    const ov::Shape shape = mask.get_shape();
    OPENVINO_ASSERT(shape[1] == 1, "Mask must have one channel");
    OPENVINO_ASSERT(factor > 0 && shape[2] % factor == 0 && shape[3] % factor == 0,
                    "Mask dimensions must be divisible by the pooling factor");

    const ov::Shape output_shape{shape[0], 1, shape[2] / factor, shape[3] / factor};
    ov::Tensor result(ov::element::f32, output_shape);
    const float* source = mask.data<const float>();
    float* destination = result.data<float>();
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        for (size_t output_y = 0; output_y < output_shape[2]; ++output_y) {
            for (size_t output_x = 0; output_x < output_shape[3]; ++output_x) {
                float maximum = 0.0f;
                for (size_t y = 0; y < factor; ++y) {
                    for (size_t x = 0; x < factor; ++x) {
                        const size_t source_index = batch * shape[2] * shape[3] +
                                                    (output_y * factor + y) * shape[3] +
                                                    output_x * factor + x;
                        maximum = std::max(maximum, source[source_index]);
                    }
                }
                destination[batch * output_shape[2] * output_shape[3] +
                            output_y * output_shape[3] + output_x] = maximum;
            }
        }
    }
    return result;
}

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
