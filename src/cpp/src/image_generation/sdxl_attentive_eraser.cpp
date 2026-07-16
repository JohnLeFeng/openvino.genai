// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sdxl_attentive_eraser.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/schedulers/ddim.hpp"
#include "sdxl_attentive_eraser_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

void validate_f32_nchw(const ov::Tensor& tensor, const char* name) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                    name,
                    " must have f32 element type");
    OPENVINO_ASSERT(tensor.get_shape().size() == 4, name, " must be a rank-4 NCHW tensor");
}

size_t reflect_index(int64_t index, size_t size) {
    OPENVINO_ASSERT(size > 1, "Reflection padding requires a spatial dimension greater than one");
    while (index < 0 || index >= static_cast<int64_t>(size)) {
        index = index < 0 ? -index : 2 * static_cast<int64_t>(size) - index - 2;
    }
    return static_cast<size_t>(index);
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
                const size_t source_index = (y * shape[2] + x) * 3 + channel;
                const size_t destination_index = channel * spatial_size + y * shape[2] + x;
                destination[destination_index] = static_cast<float>(source[source_index]) / 127.5f - 1.0f;
            }
        }
    }
    return result;
}

ov::Tensor preprocess_mask(const ov::Tensor& mask) {
    const ov::Shape& shape = mask.get_shape();
    ov::Tensor gray_mask(ov::element::f32, {1, 1, shape[1], shape[2]});
    const uint8_t* source = mask.data<const uint8_t>();
    float* destination = gray_mask.data<float>();
    const size_t channels = shape[3];
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[2]; ++x) {
            const size_t source_index = (y * shape[2] + x) * channels;
            if (channels == 1) {
                destination[y * shape[2] + x] = static_cast<float>(source[source_index]) / 255.0f;
            } else {
                destination[y * shape[2] + x] =
                    (0.299f * source[source_index] +
                     0.587f * source[source_index + 1] +
                     0.114f * source[source_index + 2]) /
                    255.0f;
            }
        }
    }
    return attentive_eraser::gaussian_blur_and_binarize_mask(gray_mask, 77, 0.1f);
}

void validate_input_tensor(const ov::Tensor& tensor,
                           const char* name,
                           bool is_mask) {
    OPENVINO_ASSERT(tensor, name, " must not be empty");
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::u8, name, " must have u8 element type");
    const ov::Shape& shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, name, " must have NHWC rank-4 shape");
    OPENVINO_ASSERT(shape[0] == 1, name, " must have batch size one");
    OPENVINO_ASSERT(shape[1] == 1024 && shape[2] == 1024,
                    name,
                    " must be 1024x1024 for the fixed attentive eraser graph");
    if (is_mask) {
        OPENVINO_ASSERT(shape[3] == 1 || shape[3] == 3, name, " must have one or three channels");
    } else {
        OPENVINO_ASSERT(shape[3] == 3, name, " must have three channels");
    }
}

}  // namespace

namespace attentive_eraser {

ov::Tensor gaussian_blur_and_binarize_mask(const ov::Tensor& gray_mask,
                                            size_t kernel_size,
                                            float threshold) {
    validate_f32_nchw(gray_mask, "Mask");
    const ov::Shape shape = gray_mask.get_shape();
    OPENVINO_ASSERT(shape[1] == 1, "Mask must have one channel");
    OPENVINO_ASSERT(kernel_size > 0 && kernel_size % 2 == 1, "Gaussian kernel size must be positive and odd");
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
                const size_t output_index = batch * output_shape[2] * output_shape[3] +
                                            output_y * output_shape[3] + output_x;
                destination[output_index] = maximum;
            }
        }
    }

    return result;
}

ov::Tensor removal_guidance(const ov::Tensor& noise_pair, float scale) {
    validate_f32_nchw(noise_pair, "Noise prediction");
    const ov::Shape shape = noise_pair.get_shape();
    OPENVINO_ASSERT(shape[0] == 2, "Noise prediction must contain the without-mask and with-mask batches");
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

void blend_latents(const ov::Tensor& initial_noised,
                   const ov::Tensor& mask,
                   ov::Tensor latents) {
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

class AttentiveEraserUNet {
public:
    explicit AttentiveEraserUNet(const std::filesystem::path& model_path) {
        model = utils::singleton_core().read_model(model_path);
        const std::set<std::string> expected_inputs{
            "sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "mask", "cur_step", "ss_steps"};
        std::set<std::string> actual_inputs;
        for (const ov::Output<ov::Node>& input : model->inputs()) {
            actual_inputs.insert(input.get_any_name());
        }
        OPENVINO_ASSERT(actual_inputs == expected_inputs,
                        "Attentive Eraser UNet inputs do not match the required six-input contract");

        model->reshape({{"sample", {2, 4, 128, 128}},
                {"timestep", {}},
                {"encoder_hidden_states", {2, 77, 2048}},
                {"text_embeds", {2, 1280}},
                {"time_ids", {2, 6}},
                {"mask", {1, 1, 1024, 1024}},
                {"cur_step", {}},
                {"ss_steps", {}}});

        ov::preprocess::PrePostProcessor preprocessor(model);
        for (const char* name : {"sample", "encoder_hidden_states", "text_embeds", "time_ids", "mask"}) {
            preprocessor.input(name).tensor().set_element_type(ov::element::f32);
        }
        preprocessor.output(0).tensor().set_element_type(ov::element::f32);
        model = preprocessor.build();
    }

    void compile(const std::string& device, const ov::AnyMap& properties) {
        OPENVINO_ASSERT(!request, "Attentive Eraser UNet is already compiled");
        request = utils::singleton_core().compile_model(model, device, properties).create_infer_request();
        model.reset();
    }

    ov::Tensor infer(const ov::Tensor& sample,
                     int64_t timestep,
                     const ov::Tensor& encoder_hidden_states,
                     const ov::Tensor& text_embeds,
                     const ov::Tensor& time_ids,
                     const ov::Tensor& mask,
                     size_t current_step,
                     size_t ss_steps) {
        OPENVINO_ASSERT(request, "Attentive Eraser UNet must be compiled before generation");
        ov::Tensor timestep_tensor(ov::element::i64, {});
        *timestep_tensor.data<int64_t>() = timestep;
        ov::Tensor current_step_tensor(ov::element::i64, {});
        *current_step_tensor.data<int64_t>() = static_cast<int64_t>(current_step);
        ov::Tensor ss_steps_tensor(ov::element::i64, {});
        *ss_steps_tensor.data<int64_t>() = static_cast<int64_t>(ss_steps);
        request.set_tensor("sample", sample);
        request.set_tensor("timestep", timestep_tensor);
        request.set_tensor("encoder_hidden_states", encoder_hidden_states);
        request.set_tensor("text_embeds", text_embeds);
        request.set_tensor("time_ids", time_ids);
        request.set_tensor("mask", mask);
        request.set_tensor("cur_step", current_step_tensor);
        request.set_tensor("ss_steps", ss_steps_tensor);
        request.infer();
        return request.get_output_tensor(0);
    }

private:
    std::shared_ptr<ov::Model> model;
    ov::InferRequest request;
};

class SDXLAttentiveEraser::Impl {
public:
    explicit Impl(std::filesystem::path models_path)
        : root_dir(std::move(models_path)),
          text_encoder(std::make_shared<CLIPTextModel>(root_dir / "text_encoder")),
          text_encoder_2(std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder_2")),
          vae(std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder")),
          scheduler(std::make_shared<DDIMScheduler>(root_dir / "scheduler" / "scheduler_config.json")),
          unet(std::make_unique<AttentiveEraserUNet>(root_dir / "unet" / "openvino_model.xml")) {}

    void compile(const std::string& device, const ov::AnyMap& properties) {
        OPENVINO_ASSERT(!compiled, "SDXLAttentiveEraser is already compiled");
        text_encoder->compile(device, properties);
        text_encoder_2->compile(device, properties);
        unet->compile(device, properties);
        vae->compile(device, properties);
        compiled = true;
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor> compute_conditioning() {
        auto inference_start = std::chrono::steady_clock::now();
        ov::Tensor text_embeds_single = text_encoder_2->infer("", "", false);
        metrics.encoder_inference_duration["text_encoder_2"] =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
        inference_start = std::chrono::steady_clock::now();
        text_encoder->infer("", "", false);
        metrics.encoder_inference_duration["text_encoder"] =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
        const size_t hidden_state_index_1 = text_encoder->get_config().num_hidden_layers + 1;
        const size_t hidden_state_index_2 = text_encoder_2->get_config().num_hidden_layers + 1;
        ov::Tensor hidden_states_single = numpy_utils::concat(
            text_encoder->get_output_tensor(hidden_state_index_1),
            text_encoder_2->get_output_tensor(hidden_state_index_2),
            -1);
        ov::Tensor hidden_states = numpy_utils::repeat(hidden_states_single, 2);
        ov::Tensor text_embeds = numpy_utils::repeat(text_embeds_single, 2);
        ov::Tensor time_ids(ov::element::f32, {2, 6});
        const std::array<float, 6> values{1024.0f, 1024.0f, 0.0f, 0.0f, 1024.0f, 1024.0f};
        std::copy(values.begin(), values.end(), time_ids.data<float>());
        std::copy(values.begin(), values.end(), time_ids.data<float>() + values.size());
        return {hidden_states, text_embeds, time_ids};
    }

    std::filesystem::path root_dir;
    SDXLAttentiveEraserConfig generation_config;
    ImageGenerationPerfMetrics metrics;
    std::shared_ptr<CLIPTextModel> text_encoder;
    std::shared_ptr<CLIPTextModelWithProjection> text_encoder_2;
    std::shared_ptr<AutoencoderKL> vae;
    std::shared_ptr<DDIMScheduler> scheduler;
    std::unique_ptr<AttentiveEraserUNet> unet;
    bool compiled = false;
};

void SDXLAttentiveEraserConfig::validate() const {
    OPENVINO_ASSERT(rm_guidance_scale > 0.0f, "rm_guidance_scale must be positive");
    OPENVINO_ASSERT(strength > 0.0f && strength <= 1.0f, "strength must be in (0, 1]");
    OPENVINO_ASSERT(num_inference_steps > 0, "num_inference_steps must be positive");
}

SDXLAttentiveEraser::SDXLAttentiveEraser(const std::filesystem::path& models_path)
    : m_impl(nullptr) {
    const auto load_start = std::chrono::steady_clock::now();
    m_impl = std::make_unique<Impl>(models_path);
    m_impl->metrics.clean_up();
    m_impl->metrics.load_time =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - load_start).count();
}

SDXLAttentiveEraser::SDXLAttentiveEraser(const std::filesystem::path& models_path,
                                         const std::string& device,
                                         const ov::AnyMap& properties)
    : SDXLAttentiveEraser(models_path) {
    compile(device, properties);
}

SDXLAttentiveEraser::~SDXLAttentiveEraser() = default;
SDXLAttentiveEraser::SDXLAttentiveEraser(SDXLAttentiveEraser&&) noexcept = default;
SDXLAttentiveEraser& SDXLAttentiveEraser::operator=(SDXLAttentiveEraser&&) noexcept = default;

void SDXLAttentiveEraser::compile(const std::string& device, const ov::AnyMap& properties) {
    m_impl->compile(device, properties);
}

SDXLAttentiveEraserConfig SDXLAttentiveEraser::get_generation_config() const {
    return m_impl->generation_config;
}

void SDXLAttentiveEraser::set_generation_config(const SDXLAttentiveEraserConfig& generation_config) {
    generation_config.validate();
    m_impl->generation_config = generation_config;
}

ov::Tensor SDXLAttentiveEraser::generate(ov::Tensor initial_image,
                                         ov::Tensor mask_image,
                                         const SDXLAttentiveEraserConfig& generation_config) {
    OPENVINO_ASSERT(m_impl->compiled, "SDXLAttentiveEraser must be compiled before generation");
    generation_config.validate();
    validate_input_tensor(initial_image, "Initial image", false);
    validate_input_tensor(mask_image, "Mask image", true);
    const float load_time = m_impl->metrics.load_time;
    m_impl->metrics.clean_up();
    m_impl->metrics.load_time = load_time;
    const auto generate_start = std::chrono::steady_clock::now();

    std::shared_ptr<Generator> generator = generation_config.generator;
    if (!generator) {
        generator = std::make_shared<CppStdGenerator>(static_cast<uint32_t>(generation_config.rng_seed));
    }

    ov::Tensor processed_image = preprocess_image(initial_image);
    ov::Tensor full_resolution_mask = preprocess_mask(mask_image);
    ov::Tensor latent_mask = attentive_eraser::max_pool_mask(full_resolution_mask, 8);
    const auto vae_encode_start = std::chrono::steady_clock::now();
    ov::Tensor image_latent = m_impl->vae->encode(processed_image, generator);
    m_impl->metrics.vae_encoder_inference_duration =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - vae_encode_start).count();
    ov::Tensor noise = generator->randn_tensor(image_latent.get_shape());

    m_impl->scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
    const std::vector<int64_t> timesteps = m_impl->scheduler->get_timesteps();
    OPENVINO_ASSERT(!timesteps.empty(), "DDIM scheduler produced no timesteps");

    ov::Tensor latent(image_latent.get_element_type(), image_latent.get_shape());
    image_latent.copy_to(latent);
    m_impl->scheduler->add_noise(latent, noise, timesteps.front());

    auto [hidden_states, text_embeds, time_ids] = m_impl->compute_conditioning();
    for (size_t step = 0; step < timesteps.size(); ++step) {
        const auto iteration_start = std::chrono::steady_clock::now();
        ov::Tensor latent_pair = numpy_utils::repeat(latent, 2);
        const auto unet_start = std::chrono::steady_clock::now();
        ov::Tensor noise_pair = m_impl->unet->infer(latent_pair,
                                                   timesteps[step],
                                                   hidden_states,
                                                   text_embeds,
                                                   time_ids,
                                                   full_resolution_mask,
                                                   step,
                                                   generation_config.ss_steps);
        m_impl->metrics.raw_metrics.unet_inference_durations.push_back(
            std::chrono::duration_cast<MicroSeconds>(std::chrono::steady_clock::now() - unet_start));
        ov::Tensor guided_noise = attentive_eraser::removal_guidance(
            noise_pair, generation_config.rm_guidance_scale);
        latent = m_impl->scheduler->step(guided_noise, latent, step, generator).at("latent");

        ov::Tensor initial_noised(image_latent.get_element_type(), image_latent.get_shape());
        image_latent.copy_to(initial_noised);
        if (step + 1 < timesteps.size()) {
            m_impl->scheduler->add_noise(initial_noised, noise, timesteps[step + 1]);
        }
        attentive_eraser::blend_latents(initial_noised, latent_mask, latent);
        m_impl->metrics.raw_metrics.iteration_durations.push_back(
            std::chrono::duration_cast<MicroSeconds>(std::chrono::steady_clock::now() - iteration_start));

        if (generation_config.callback &&
            generation_config.callback(step, timesteps.size(), latent)) {
            break;
        }
    }

    const auto vae_decode_start = std::chrono::steady_clock::now();
    ov::Tensor result = m_impl->vae->decode(latent);
    m_impl->metrics.vae_decoder_inference_duration =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - vae_decode_start).count();
    m_impl->metrics.generate_duration =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - generate_start).count();
    return result;
}

ov::Tensor SDXLAttentiveEraser::generate(ov::Tensor initial_image, ov::Tensor mask_image) {
    return generate(std::move(initial_image), std::move(mask_image), m_impl->generation_config);
}

ov::Tensor SDXLAttentiveEraser::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_impl->compiled, "SDXLAttentiveEraser must be compiled before decoding");
    return m_impl->vae->decode(latent);
}

ImageGenerationPerfMetrics SDXLAttentiveEraser::get_performance_metrics() {
    m_impl->metrics.evaluate_statistics();
    return m_impl->metrics;
}

}  // namespace genai
}  // namespace ov