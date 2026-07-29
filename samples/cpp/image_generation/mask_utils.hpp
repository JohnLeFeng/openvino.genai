// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace mask_utils {

inline size_t reflect_index(int64_t index, size_t size) {
    if (size <= 1) {
        throw std::runtime_error("Reflection padding requires a spatial dimension greater than one");
    }
    while (index < 0 || index >= static_cast<int64_t>(size)) {
        index = index < 0 ? -index : 2 * static_cast<int64_t>(size) - index - 2;
    }
    return static_cast<size_t>(index);
}

/**
 * Apply Gaussian blur and binarization to a mask.
 * @param gray_mask Input mask tensor with shape [batch, 1, height, width], element type f32
 * @param kernel_size Gaussian kernel size (must be positive and odd)
 * @param threshold Binarization threshold in range [0, 1]
 * @return Processed mask tensor with values 0.0 (no mask) or 1.0 (mask)
 *
 * Example usage:
 *   ov::Tensor processed_mask = mask_utils::gaussian_blur_and_binarize_mask(input_mask, 77, 0.1f);
 */
inline ov::Tensor gaussian_blur_and_binarize_mask(const ov::Tensor& gray_mask,
                                                  size_t kernel_size,
                                                  float threshold) {
    // Validate inputs
    if (!gray_mask) {
        throw std::invalid_argument("Mask tensor must not be empty");
    }
    if (gray_mask.get_element_type() != ov::element::f32) {
        throw std::invalid_argument("Mask must have f32 element type");
    }
    if (gray_mask.get_shape().size() != 4) {
        throw std::invalid_argument("Mask must be a rank-4 NCHW tensor");
    }

    const ov::Shape shape = gray_mask.get_shape();
    if (shape[1] != 1) {
        throw std::invalid_argument("Mask must have one channel");
    }
    if (kernel_size == 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument("Gaussian kernel size must be positive and odd");
    }
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Mask threshold must be in [0, 1]");
    }

    // Compute Gaussian kernel
    const float sigma = 0.3f * ((static_cast<float>(kernel_size) - 1.0f) * 0.5f - 1.0f) + 0.8f;
    const int64_t radius = static_cast<int64_t>(kernel_size / 2);
    std::vector<float> kernel(kernel_size);
    for (size_t index = 0; index < kernel_size; ++index) {
        const float distance = static_cast<float>(static_cast<int64_t>(index) - radius);
        kernel[index] = std::exp(-(distance * distance) / (2.0f * sigma * sigma));
    }
    const float kernel_sum = std::accumulate(kernel.begin(), kernel.end(), 0.0f);
    for (float& value : kernel) {
        value /= kernel_sum;
    }

    // Apply horizontal and vertical Gaussian blur
    ov::Tensor horizontal(ov::element::f32, shape);
    ov::Tensor result(ov::element::f32, shape);
    const float* source = gray_mask.data<const float>();
    float* horizontal_data = horizontal.data<float>();
    float* result_data = result.data<float>();
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const size_t batch_offset = batch * height * width;

        // Horizontal pass
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_x =
                        reflect_index(static_cast<int64_t>(x) + static_cast<int64_t>(kernel_index) - radius, width);
                    value += source[batch_offset + y * width + source_x] * kernel[kernel_index];
                }
                horizontal_data[batch_offset + y * width + x] = value;
            }
        }

        // Vertical pass with binarization
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_y = reflect_index(static_cast<int64_t>(y) + static_cast<int64_t>(kernel_index) - radius, height);
                    value += horizontal_data[batch_offset + source_y * width + x] * kernel[kernel_index];
                }
                result_data[batch_offset + y * width + x] = value < threshold ? 0.0f : 1.0f;
            }
        }
    }

    return result;
}

/**
 * Apply max pooling to a mask.
 * @param mask Input mask tensor with shape [batch, 1, height, width], element type f32
 * @param factor Pooling factor (height and width must be divisible by factor)
 * @return Pooled mask tensor with shape [batch, 1, height/factor, width/factor]
 *
 * Example usage:
 *   ov::Tensor pooled_mask = mask_utils::max_pool_mask(blurred_mask, 8);
 */
inline ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor) {
    if (!mask) {
        throw std::invalid_argument("Mask tensor must not be empty");
    }
    if (mask.get_element_type() != ov::element::f32) {
        throw std::invalid_argument("Mask must have f32 element type");
    }
    if (mask.get_shape().size() != 4) {
        throw std::invalid_argument("Mask must be a rank-4 NCHW tensor");
    }

    const ov::Shape shape = mask.get_shape();
    if (shape[1] != 1) {
        throw std::invalid_argument("Mask must have one channel");
    }
    if (factor == 0 || shape[2] % factor != 0 || shape[3] % factor != 0) {
        throw std::invalid_argument("Mask dimensions must be divisible by the pooling factor");
    }

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
                        const size_t source_index = batch * shape[2] * shape[3] + (output_y * factor + y) * shape[3] +
                                                    output_x * factor + x;
                        maximum = std::max(maximum, source[source_index]);
                    }
                }
                const size_t output_index = batch * output_shape[2] * output_shape[3] + output_y * output_shape[3] + output_x;
                destination[output_index] = maximum;
            }
        }
    }

    return result;
}

}  // namespace mask_utils
