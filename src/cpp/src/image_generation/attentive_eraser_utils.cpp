// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/attentive_eraser_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ov {
namespace genai {

namespace {

void validate_f32_nchw(const ov::Tensor& tensor, const char* name) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32, name, " must have f32 element type");
    OPENVINO_ASSERT(tensor.get_shape().size() == 4, name, " must be a rank-4 NCHW tensor");
}

size_t reflect_index(int64_t index, size_t size) {
    OPENVINO_ASSERT(size > 1, "Reflection padding requires a spatial dimension greater than one");
    while (index < 0 || index >= static_cast<int64_t>(size)) {
        index = index < 0 ? -index : 2 * static_cast<int64_t>(size) - index - 2;
    }
    return static_cast<size_t>(index);
}

ov::Tensor gaussian_blur_and_binarize_mask(const ov::Tensor& gray_mask,
                                            size_t kernel_size,
                                            float threshold) {
    validate_f32_nchw(gray_mask, "Mask");
    const ov::Shape shape = gray_mask.get_shape();
    OPENVINO_ASSERT(shape[1] == 1, "Mask must have one channel");
    OPENVINO_ASSERT(kernel_size > 0 && kernel_size % 2 == 1,
                    "Gaussian kernel size must be positive and odd");
    OPENVINO_ASSERT(threshold >= 0.0f && threshold <= 1.0f, "Mask threshold must be in [0, 1]");

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

    ov::Tensor horizontal(ov::element::f32, shape);
    ov::Tensor result(ov::element::f32, shape);
    const float* source = gray_mask.data<const float>();
    float* horizontal_data = horizontal.data<float>();
    float* result_data = result.data<float>();
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const size_t batch_offset = batch * height * width;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_x = reflect_index(static_cast<int64_t>(x) +
                                                              static_cast<int64_t>(kernel_index) - radius,
                                                          width);
                    value += source[batch_offset + y * width + source_x] * kernel[kernel_index];
                }
                horizontal_data[batch_offset + y * width + x] = value;
            }
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_y = reflect_index(static_cast<int64_t>(y) +
                                                              static_cast<int64_t>(kernel_index) - radius,
                                                          height);
                    value += horizontal_data[batch_offset + source_y * width + x] * kernel[kernel_index];
                }
                result_data[batch_offset + y * width + x] = value < threshold ? 0.0f : 1.0f;
            }
        }
    }
    return result;
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

ov::Tensor preprocess_attentive_mask(const ov::Tensor& mask,
                                      size_t kernel_size,
                                      float threshold) {
    const ov::Shape& shape = mask.get_shape();
    OPENVINO_ASSERT(mask.get_element_type() == ov::element::u8, "Mask must have u8 element type");
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1, "Mask must be rank-4 NHWC with batch 1");
    const size_t channels = shape[3];
    OPENVINO_ASSERT(channels == 1 || channels == 3, "Mask must have 1 or 3 channels");

    ov::Tensor gray_mask(ov::element::f32, {1, 1, shape[1], shape[2]});
    const uint8_t* source = mask.data<const uint8_t>();
    float* destination = gray_mask.data<float>();
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[2]; ++x) {
            const size_t source_index = (y * shape[2] + x) * channels;
            destination[y * shape[2] + x] = channels == 1
                ? static_cast<float>(source[source_index]) / 255.0f
                : (0.299f * source[source_index] + 0.587f * source[source_index + 1] +
                   0.114f * source[source_index + 2]) / 255.0f;
        }
    }
    return gaussian_blur_and_binarize_mask(gray_mask, kernel_size, threshold);
}

}  // namespace genai
}  // namespace ov
