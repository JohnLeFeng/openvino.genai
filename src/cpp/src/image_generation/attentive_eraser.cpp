// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "attentive_eraser.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

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

}  // namespace

void validate_attentive_eraser_unet_inputs(const std::shared_ptr<ov::Model>& model,
                                           AttentiveEraserModelFamily model_family,
                                           bool attentive_eraser_enabled) {
    OPENVINO_ASSERT(model, "UNet model must not be null");

    std::set<std::string> actual_inputs;
    for (const auto& input : model->inputs()) {
        actual_inputs.insert(input.get_any_name());
    }

    const bool has_attentive_inputs = actual_inputs.count("mask") > 0 ||
                                      actual_inputs.count("cur_step") > 0 ||
                                      actual_inputs.count("ss_steps") > 0;
    if (!attentive_eraser_enabled) {
        OPENVINO_ASSERT(!has_attentive_inputs,
                        "Attentive Eraser UNet inputs require the attentive inpainting mode constructor property");
        return;
    }

    std::set<std::string> expected_inputs{
        "sample", "timestep", "encoder_hidden_states", "mask", "cur_step", "ss_steps"};
    if (model_family == AttentiveEraserModelFamily::STABLE_DIFFUSION_XL) {
        expected_inputs.insert("text_embeds");
        expected_inputs.insert("time_ids");
    }

    OPENVINO_ASSERT(actual_inputs == expected_inputs,
                    model_family == AttentiveEraserModelFamily::STABLE_DIFFUSION_XL ? "SDXL" : "SD",
                    " Attentive Eraser UNet inputs do not match the required ",
                    expected_inputs.size(),
                    "-input contract");
}

std::shared_ptr<ov::Model> prepare_attentive_eraser_unet_model(std::shared_ptr<ov::Model> model) {
    OPENVINO_ASSERT(model, "UNet model must not be null");

    std::set<std::string> input_names;
    for (const auto& input : model->inputs()) {
        input_names.insert(input.get_any_name());
    }
    if (input_names.count("mask") == 0 || input_names.count("cur_step") == 0 ||
        input_names.count("ss_steps") == 0) {
        return model;
    }

    ov::preprocess::PrePostProcessor preprocessor(model);
    for (const char* input_name : {"sample", "encoder_hidden_states", "text_embeds", "time_ids", "mask"}) {
        if (input_names.count(input_name) > 0) {
            preprocessor.input(input_name).tensor().set_element_type(ov::element::f32);
        }
    }
    preprocessor.output(0).tensor().set_element_type(ov::element::f32);
    return preprocessor.build();
}

void reshape_attentive_eraser_unet_model(const std::shared_ptr<ov::Model>& model,
                                         size_t sample_size,
                                         size_t vae_scale_factor,
                                         size_t cross_attention_dim) {
    OPENVINO_ASSERT(model, "UNet model must not be null");

    std::set<std::string> input_names;
    for (const auto& input : model->inputs()) {
        input_names.insert(input.get_any_name());
    }
    if (input_names.count("mask") == 0 || input_names.count("cur_step") == 0 ||
        input_names.count("ss_steps") == 0) {
        return;
    }

    OPENVINO_ASSERT(sample_size > 0 && vae_scale_factor > 0 && cross_attention_dim > 0,
                    "Attentive Eraser UNet requires static sample, VAE scale, and cross-attention dimensions");
    const size_t image_size = sample_size * vae_scale_factor;
    std::map<std::string, ov::PartialShape> shapes{
        {"sample", {2, 4, sample_size, sample_size}},
        {"timestep", {}},
        {"encoder_hidden_states", {2, 77, cross_attention_dim}},
        {"mask", {1, 1, image_size, image_size}},
        {"cur_step", {}},
        {"ss_steps", {}}};
    if (input_names.count("text_embeds") > 0) {
        shapes["text_embeds"] = {2, 1280};
        shapes["time_ids"] = {2, 6};
    }
    model->reshape(shapes);
}

std::shared_ptr<Scheduler> create_attentive_eraser_scheduler(
    const std::filesystem::path& scheduler_config_path,
    bool attentive_eraser_enabled) {
    return Scheduler::from_config(scheduler_config_path,
                                  attentive_eraser_enabled ? Scheduler::Type::DDIM : Scheduler::Type::AUTO);
}

namespace attentive_eraser {

void validate_input_tensor(const ov::Tensor& tensor,
                           const char* name,
                           bool is_mask,
                           size_t image_size) {
    OPENVINO_ASSERT(tensor, name, " must not be empty");
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::u8, name, " must have u8 element type");
    const ov::Shape& shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, name, " must have NHWC rank-4 shape");
    OPENVINO_ASSERT(shape[0] == 1, name, " must have batch size one");
    OPENVINO_ASSERT(shape[1] == image_size && shape[2] == image_size,
                    name,
                    " must be ", image_size, "x", image_size, " for the fixed attentive eraser graph");
    OPENVINO_ASSERT(is_mask ? shape[3] == 1 || shape[3] == 3 : shape[3] == 3,
                    name,
                    is_mask ? " must have one or three channels" : " must have three channels");
}

ov::Tensor preprocess_image(const ov::Tensor& image) {
    const ov::Shape& shape = image.get_shape();
    ov::Tensor result(ov::element::f32, {1, 3, shape[1], shape[2]});
    const uint8_t* source = image.data<const uint8_t>();
    float* destination = result.data<float>();
    const size_t spatial_size = shape[1] * shape[2];
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[2]; ++x) {
            for (size_t channel = 0; channel < 3; ++channel) {
                destination[channel * spatial_size + y * shape[2] + x] =
                    static_cast<float>(source[(y * shape[2] + x) * 3 + channel]) / 127.5f - 1.0f;
            }
        }
    }
    return result;
}

ov::Tensor preprocess_mask(const ov::Tensor& mask, size_t kernel_size, float threshold) {
    const ov::Shape& shape = mask.get_shape();
    ov::Tensor gray_mask(ov::element::f32, {1, 1, shape[1], shape[2]});
    const uint8_t* source = mask.data<const uint8_t>();
    float* destination = gray_mask.data<float>();
    const size_t channels = shape[3];
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

ov::Tensor removal_guidance(const ov::Tensor& noise_pair, float scale) {
    validate_f32_nchw(noise_pair, "Noise prediction");
    const ov::Shape shape = noise_pair.get_shape();
    OPENVINO_ASSERT(shape[0] == 2,
                    "Noise prediction must contain the without-mask and with-mask batches");
    OPENVINO_ASSERT(scale > 0.0f, "Removal guidance scale must be positive");

    ov::Shape output_shape = shape;
    output_shape[0] = 1;
    ov::Tensor result(ov::element::f32, output_shape);
    const float* without_mask = noise_pair.data<const float>();
    const float* with_mask = without_mask + result.get_size();
    float* destination = result.data<float>();
    for (size_t index = 0; index < result.get_size(); ++index) {
        destination[index] = without_mask[index] + scale * (with_mask[index] - without_mask[index]);
    }
    return result;
}

void blend_latents(const ov::Tensor& initial_noised, const ov::Tensor& mask, ov::Tensor latents) {
    validate_f32_nchw(initial_noised, "Initial noised latent");
    validate_f32_nchw(mask, "Latent mask");
    validate_f32_nchw(latents, "Latent");
    OPENVINO_ASSERT(initial_noised.get_shape() == latents.get_shape(),
                    "Initial noised latent and latent shapes must match");
    const ov::Shape latent_shape = latents.get_shape();
    const ov::Shape mask_shape = mask.get_shape();
    OPENVINO_ASSERT(mask_shape[0] == latent_shape[0] && mask_shape[1] == 1 &&
                        mask_shape[2] == latent_shape[2] && mask_shape[3] == latent_shape[3],
                    "Latent mask must have shape [batch, 1, height, width]");

    const float* initial_data = initial_noised.data<const float>();
    const float* mask_data = mask.data<const float>();
    float* latent_data = latents.data<float>();
    const size_t spatial_size = latent_shape[2] * latent_shape[3];
    for (size_t batch = 0; batch < latent_shape[0]; ++batch) {
        for (size_t channel = 0; channel < latent_shape[1]; ++channel) {
            for (size_t spatial = 0; spatial < spatial_size; ++spatial) {
                const size_t latent_index = (batch * latent_shape[1] + channel) * spatial_size + spatial;
                const float mask_value = mask_data[batch * spatial_size + spatial];
                latent_data[latent_index] = (1.0f - mask_value) * initial_data[latent_index] +
                                            mask_value * latent_data[latent_index];
            }
        }
    }
}

}  // namespace attentive_eraser

}  // namespace genai
}  // namespace ov