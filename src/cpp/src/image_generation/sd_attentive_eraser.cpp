// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sd_attentive_eraser.hpp"

#include <algorithm>
#include <chrono>
#include <set>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/schedulers/ddim.hpp"
#include "sdxl_attentive_eraser_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

constexpr size_t image_size = 512;
constexpr size_t latent_size = 64;

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

ov::Tensor preprocess_mask(const ov::Tensor& mask) {
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
    return attentive_eraser::gaussian_blur_and_binarize_mask(gray_mask, 7, 0.1f);
}

void validate_input_tensor(const ov::Tensor& tensor, const char* name, bool is_mask) {
    OPENVINO_ASSERT(tensor, name, " must not be empty");
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::u8, name, " must have u8 element type");
    const ov::Shape& shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, name, " must have NHWC rank-4 shape");
    OPENVINO_ASSERT(shape[0] == 1, name, " must have batch size one");
    OPENVINO_ASSERT(shape[1] == image_size && shape[2] == image_size,
                    name,
                    " must be 512x512 for the fixed attentive eraser graph");
    if (is_mask) {
        OPENVINO_ASSERT(shape[3] == 1 || shape[3] == 3, name, " must have one or three channels");
    } else {
        OPENVINO_ASSERT(shape[3] == 3, name, " must have three channels");
    }
}

class AttentiveEraserUNet {
public:
    AttentiveEraserUNet(const std::filesystem::path& model_path, size_t hidden_size) {
        model = utils::singleton_core().read_model(model_path);
        const std::set<std::string> expected_inputs{
            "sample", "timestep", "encoder_hidden_states", "mask", "cur_step", "ss_steps"};
        std::set<std::string> actual_inputs;
        for (const ov::Output<ov::Node>& input : model->inputs()) {
            actual_inputs.insert(input.get_any_name());
        }
        OPENVINO_ASSERT(actual_inputs == expected_inputs,
                        "SD Attentive Eraser UNet inputs do not match the required six-input contract");

        model->reshape({{"sample", {2, 4, latent_size, latent_size}},
                        {"timestep", {}},
                        {"encoder_hidden_states", {2, 77, hidden_size}},
                        {"mask", {1, 1, image_size, image_size}},
                        {"cur_step", {}},
                        {"ss_steps", {}}});

        ov::preprocess::PrePostProcessor preprocessor(model);
        for (const char* name : {"sample", "encoder_hidden_states", "mask"}) {
            preprocessor.input(name).tensor().set_element_type(ov::element::f32);
        }
        preprocessor.output(0).tensor().set_element_type(ov::element::f32);
        model = preprocessor.build();
    }

    void compile(const std::string& device, const ov::AnyMap& properties) {
        OPENVINO_ASSERT(!request, "SD Attentive Eraser UNet is already compiled");
        request = utils::singleton_core().compile_model(model, device, properties).create_infer_request();
        model.reset();
    }

    ov::Tensor infer(const ov::Tensor& sample,
                     int64_t timestep,
                     const ov::Tensor& encoder_hidden_states,
                     const ov::Tensor& mask,
                     size_t current_step,
                     size_t ss_steps) {
        OPENVINO_ASSERT(request, "SD Attentive Eraser UNet must be compiled before generation");
        ov::Tensor timestep_tensor(ov::element::i64, {});
        *timestep_tensor.data<int64_t>() = timestep;
        ov::Tensor current_step_tensor(ov::element::i64, {});
        *current_step_tensor.data<int64_t>() = static_cast<int64_t>(current_step);
        ov::Tensor ss_steps_tensor(ov::element::i64, {});
        *ss_steps_tensor.data<int64_t>() = static_cast<int64_t>(ss_steps);
        request.set_tensor("sample", sample);
        request.set_tensor("timestep", timestep_tensor);
        request.set_tensor("encoder_hidden_states", encoder_hidden_states);
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

}  // namespace

void SDAttentiveEraserConfig::validate() const {
    OPENVINO_ASSERT(rm_guidance_scale > 0.0f, "rm_guidance_scale must be positive");
    OPENVINO_ASSERT(strength > 0.0f && strength <= 1.0f, "strength must be in (0, 1]");
    OPENVINO_ASSERT(num_inference_steps > 0, "num_inference_steps must be positive");
}

namespace detail {

class SDAttentiveEraserImpl {
public:
    SDAttentiveEraserImpl(std::filesystem::path models_path, size_t expected_hidden_size)
        : root_dir(std::move(models_path)),
          hidden_size(expected_hidden_size),
          text_encoder(std::make_shared<CLIPTextModel>(root_dir / "text_encoder")),
          vae(std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder")),
          scheduler(std::make_shared<DDIMScheduler>(root_dir / "scheduler" / "scheduler_config.json")),
          unet(std::make_unique<AttentiveEraserUNet>(root_dir / "unet" / "openvino_model.xml", hidden_size)) {}

    void compile(const std::string& device, const ov::AnyMap& properties) {
        OPENVINO_ASSERT(!compiled, "SD Attentive Eraser is already compiled");
        text_encoder->compile(device, properties);
        unet->compile(device, properties);
        vae->compile(device, properties);
        compiled = true;
    }

    ov::Tensor compute_conditioning() {
        const auto inference_start = std::chrono::steady_clock::now();
        ov::Tensor hidden_states_single = text_encoder->infer("", "", false);
        metrics.encoder_inference_duration["text_encoder"] =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
        const ov::Shape& shape = hidden_states_single.get_shape();
        OPENVINO_ASSERT(shape == ov::Shape({1, 77, hidden_size}),
                        "Text encoder output must have shape [1, 77, ",
                        hidden_size,
                        "] for the selected attentive eraser");
        return numpy_utils::repeat(hidden_states_single, 2);
    }

    ov::Tensor generate(ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const SDAttentiveEraserConfig& config) {
        OPENVINO_ASSERT(compiled, "SD Attentive Eraser must be compiled before generation");
        config.validate();
        validate_input_tensor(initial_image, "Initial image", false);
        validate_input_tensor(mask_image, "Mask image", true);
        const float load_time = metrics.load_time;
        metrics.clean_up();
        metrics.load_time = load_time;
        const auto generate_start = std::chrono::steady_clock::now();

        std::shared_ptr<Generator> generator = config.generator;
        if (!generator) {
            generator = std::make_shared<CppStdGenerator>(static_cast<uint32_t>(config.rng_seed));
        }

        ov::Tensor processed_image = preprocess_image(initial_image);
        ov::Tensor full_resolution_mask = preprocess_mask(mask_image);
        ov::Tensor latent_mask = attentive_eraser::max_pool_mask(full_resolution_mask, 8);
        const auto vae_encode_start = std::chrono::steady_clock::now();
        ov::Tensor image_latent = vae->encode(processed_image, generator);
        metrics.vae_encoder_inference_duration =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - vae_encode_start).count();
        ov::Tensor noise = generator->randn_tensor(image_latent.get_shape());

        scheduler->set_timesteps(config.num_inference_steps, config.strength);
        const std::vector<int64_t> timesteps = scheduler->get_timesteps();
        OPENVINO_ASSERT(!timesteps.empty(), "DDIM scheduler produced no timesteps");

        ov::Tensor latent(image_latent.get_element_type(), image_latent.get_shape());
        image_latent.copy_to(latent);
        scheduler->add_noise(latent, noise, timesteps.front());

        ov::Tensor hidden_states = compute_conditioning();
        for (size_t step = 0; step < timesteps.size(); ++step) {
            const auto iteration_start = std::chrono::steady_clock::now();
            ov::Tensor latent_pair = numpy_utils::repeat(latent, 2);
            const auto unet_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pair = unet->infer(latent_pair,
                                                timesteps[step],
                                                hidden_states,
                                                full_resolution_mask,
                                                step,
                                                config.ss_steps);
            metrics.raw_metrics.unet_inference_durations.push_back(
                std::chrono::duration_cast<MicroSeconds>(std::chrono::steady_clock::now() - unet_start));
            ov::Tensor guided_noise = attentive_eraser::removal_guidance(noise_pair, config.rm_guidance_scale);
            latent = scheduler->step(guided_noise, latent, step, generator).at("latent");

            ov::Tensor initial_noised(image_latent.get_element_type(), image_latent.get_shape());
            image_latent.copy_to(initial_noised);
            if (step + 1 < timesteps.size()) {
                scheduler->add_noise(initial_noised, noise, timesteps[step + 1]);
            }
            attentive_eraser::blend_latents(initial_noised, latent_mask, latent);
            metrics.raw_metrics.iteration_durations.push_back(
                std::chrono::duration_cast<MicroSeconds>(std::chrono::steady_clock::now() - iteration_start));

            if (config.callback && config.callback(step, timesteps.size(), latent)) {
                break;
            }
        }

        const auto vae_decode_start = std::chrono::steady_clock::now();
        ov::Tensor result = vae->decode(latent);
        metrics.vae_decoder_inference_duration =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - vae_decode_start).count();
        metrics.generate_duration =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - generate_start).count();
        return result;
    }

    std::filesystem::path root_dir;
    size_t hidden_size;
    SDAttentiveEraserConfig generation_config;
    ImageGenerationPerfMetrics metrics;
    std::shared_ptr<CLIPTextModel> text_encoder;
    std::shared_ptr<AutoencoderKL> vae;
    std::shared_ptr<DDIMScheduler> scheduler;
    std::unique_ptr<AttentiveEraserUNet> unet;
    bool compiled = false;
};

}  // namespace detail

#define OPENVINO_GENAI_DEFINE_SD_ATTENTIVE_ERASER(ClassName, HiddenSize)                                      \
    ClassName::ClassName(const std::filesystem::path& models_path)                                            \
        : m_impl(nullptr) {                                                                                   \
        const auto load_start = std::chrono::steady_clock::now();                                             \
        m_impl = std::make_unique<detail::SDAttentiveEraserImpl>(models_path, HiddenSize);                    \
        m_impl->metrics.clean_up();                                                                           \
        m_impl->metrics.load_time = std::chrono::duration<float, std::milli>(                                \
            std::chrono::steady_clock::now() - load_start).count();                                           \
    }                                                                                                         \
    ClassName::ClassName(const std::filesystem::path& models_path,                                            \
                         const std::string& device,                                                           \
                         const ov::AnyMap& properties)                                                        \
        : ClassName(models_path) {                                                                            \
        compile(device, properties);                                                                          \
    }                                                                                                         \
    ClassName::~ClassName() = default;                                                                         \
    ClassName::ClassName(ClassName&&) noexcept = default;                                                      \
    ClassName& ClassName::operator=(ClassName&&) noexcept = default;                                           \
    void ClassName::compile(const std::string& device, const ov::AnyMap& properties) {                         \
        m_impl->compile(device, properties);                                                                   \
    }                                                                                                         \
    SDAttentiveEraserConfig ClassName::get_generation_config() const {                                        \
        return m_impl->generation_config;                                                                      \
    }                                                                                                         \
    void ClassName::set_generation_config(const SDAttentiveEraserConfig& config) {                            \
        config.validate();                                                                                     \
        m_impl->generation_config = config;                                                                    \
    }                                                                                                         \
    ov::Tensor ClassName::generate(ov::Tensor initial_image,                                                  \
                                   ov::Tensor mask_image,                                                     \
                                   const SDAttentiveEraserConfig& config) {                                   \
        return m_impl->generate(std::move(initial_image), std::move(mask_image), config);                     \
    }                                                                                                         \
    ov::Tensor ClassName::generate(ov::Tensor initial_image, ov::Tensor mask_image) {                         \
        return generate(std::move(initial_image), std::move(mask_image), m_impl->generation_config);          \
    }                                                                                                         \
    ov::Tensor ClassName::decode(const ov::Tensor& latent) {                                                  \
        OPENVINO_ASSERT(m_impl->compiled, "SD Attentive Eraser must be compiled before decoding");          \
        return m_impl->vae->decode(latent);                                                                    \
    }                                                                                                         \
    ImageGenerationPerfMetrics ClassName::get_performance_metrics() {                                         \
        m_impl->metrics.evaluate_statistics();                                                                 \
        return m_impl->metrics;                                                                                \
    }

OPENVINO_GENAI_DEFINE_SD_ATTENTIVE_ERASER(SD15AttentiveEraser, 768)
OPENVINO_GENAI_DEFINE_SD_ATTENTIVE_ERASER(SD2AttentiveEraser, 1024)

#undef OPENVINO_GENAI_DEFINE_SD_ATTENTIVE_ERASER

}  // namespace genai
}  // namespace ov