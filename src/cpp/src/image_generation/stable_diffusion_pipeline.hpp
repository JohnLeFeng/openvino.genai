// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <filesystem>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"

#include "openvino/runtime/core.hpp"

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "numpy_utils.hpp"
#include "schedulers/ddim.hpp"

namespace ov {
namespace genai {

class StableDiffusionPipeline : public DiffusionPipeline {
public:
    explicit StableDiffusionPipeline(PipelineType pipeline_type) :
        DiffusionPipeline(pipeline_type) {}

    StableDiffusionPipeline(PipelineType pipeline_type,
                            const std::filesystem::path& root_dir,
                            bool use_attentive_eraser = false) :
        StableDiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
        m_use_attentive_eraser = use_attentive_eraser;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(create_attentive_eraser_scheduler(root_dir / "scheduler/scheduler_config.json",
                                m_use_attentive_eraser));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet");
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
        initialize_attentive_eraser_generation_config();
    }

    StableDiffusionPipeline(PipelineType pipeline_type,
                            const std::filesystem::path& root_dir,
                            const std::string& device,
                            const ov::AnyMap& properties,
                            bool use_attentive_eraser = false) :
        StableDiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
        m_use_attentive_eraser = use_attentive_eraser;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(create_attentive_eraser_scheduler(root_dir / "scheduler/scheduler_config.json",
                                m_use_attentive_eraser));

        auto updated_properties = update_adapters_in_properties(properties, &DiffusionPipeline::derived_adapters);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, *updated_properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, *updated_properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
        initialize_attentive_eraser_generation_config();

        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    StableDiffusionPipeline(
        PipelineType pipeline_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae,
        bool use_attentive_eraser = false)
        : StableDiffusionPipeline(pipeline_type) {
        m_use_attentive_eraser = use_attentive_eraser;
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(clip_text_model);
        m_unet = std::make_shared<UNet2DConditionModel>(unet);
        m_vae = std::make_shared<AutoencoderKL>(vae);

        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "LatentConsistencyModelPipeline" : "StableDiffusionPipeline";
        initialize_generation_config(pipeline_name);
    }

    StableDiffusionPipeline(PipelineType pipeline_type, const StableDiffusionPipeline& pipe) :
        DiffusionPipeline(pipeline_type),
        m_use_attentive_eraser(pipe.m_use_attentive_eraser) {
        OPENVINO_ASSERT(!pipe.m_use_attentive_eraser,
                "Cannot convert an Attentive Eraser inpainting pipeline to another pipeline type");
        m_root_dir = pipe.m_root_dir;
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(*pipe.m_clip_text_encoder);
        m_unet = std::make_shared<UNet2DConditionModel>(*pipe.m_unet);
        m_vae = std::make_shared<AutoencoderKL>(*pipe.m_vae);
        m_pipeline_type = pipeline_type;
        m_generation_config = pipe.m_generation_config;
        m_scheduler = pipe.m_scheduler;

        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "LatentConsistencyModelPipeline" : "StableDiffusionPipeline";
        initialize_generation_config(pipeline_name);
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
        const std::string& denoise_device,
        const std::string& vae_device,
        const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &DiffusionPipeline::derived_adapters);

        m_clip_text_encoder->compile(text_encode_device, *updated_properties);
        m_unet->compile(denoise_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);

    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<CLIPTextModel> clip_text_encoder = m_clip_text_encoder->clone();
        std::shared_ptr<UNet2DConditionModel> unet = std::make_shared<UNet2DConditionModel>(m_unet->clone());
        std::shared_ptr<StableDiffusionPipeline> pipeline = std::make_shared<StableDiffusionPipeline>(
            m_pipeline_type,
            *clip_text_encoder,
            *unet,
            *vae,
            m_use_attentive_eraser);

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(create_attentive_eraser_scheduler(m_root_dir / "scheduler/scheduler_config.json",
                                      m_use_attentive_eraser));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG

        std::string negative_prompt = generation_config.negative_prompt != std::nullopt ? *generation_config.negative_prompt : std::string{};
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor encoder_hidden_states = m_clip_text_encoder->infer(positive_prompt, negative_prompt,
            batch_size_multiplier > 1);
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

        // replicate encoder hidden state to UNet model
        if (generation_config.num_images_per_prompt == 1) {
            // reuse output of text encoder directly w/o extra memory copy
            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);
        } else {
            ov::Shape enc_shape = encoder_hidden_states.get_shape();
            enc_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor encoder_hidden_states_repeated(encoder_hidden_states.get_element_type(), enc_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);
        }

        if (unet_config.time_cond_proj_dim >= 0) { // LCM
            ov::Tensor timestep_cond = get_guidance_scale_embedding(generation_config.guidance_scale - 1.0f, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", timestep_cond);
        }
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        std::vector<int64_t> timesteps = m_scheduler->get_timesteps();
        OPENVINO_ASSERT(!timesteps.empty(), "Timesteps are not computed yet");
        int64_t latent_timestep = timesteps.front();

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const bool is_inpainting = m_pipeline_type == PipelineType::INPAINTING,
            is_strength_max = is_inpainting && generation_config.strength == 1.0f,
            return_image_latent = is_inpainting && !is_inpainting_model();

        ov::Shape latent_shape{generation_config.num_images_per_prompt, m_vae->get_config().latent_channels,
                               generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
        ov::Tensor latent(ov::element::f32, {}), proccesed_image, image_latent, noise;

        if (initial_image) {
            proccesed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
            proccesed_image = m_image_processor->execute(proccesed_image);

            // prepare image latent for cases:
            // - image to image
            // - inpainting with strength < 1.0
            // - inpainting with non-specialized model
            if (!is_strength_max || return_image_latent) {
                auto encode_start = std::chrono::steady_clock::now();
                image_latent = m_vae->encode(proccesed_image, generation_config.generator);
                m_perf_metrics.vae_encoder_inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                    std::chrono::steady_clock::now() - encode_start)
                                                                    .count();
                // in case of image to image or inpaining with strength < 1.0, we need to initialize initial latent with
                // image_latent
                if (!is_strength_max) {
                    image_latent.copy_to(latent);
                    latent = numpy_utils::repeat(latent, generation_config.num_images_per_prompt);
                }
            }
        }

        noise = generation_config.generator->randn_tensor(latent_shape);

        if (!latent.get_shape().empty()) {
            m_scheduler->add_noise(latent, noise, latent_timestep);
        } else {
            latent.set_shape(latent_shape);

            // if pure noise then scale the initial latents by the  Scheduler's init sigma
            const float * noise_data = noise.data<const float>();
            float * latent_data = latent.data<float>();
            for (size_t i = 0; i < latent.get_size(); ++i)
                latent_data[i] = noise_data[i] * m_scheduler->get_init_noise_sigma();
        }

        return std::make_tuple(latent, proccesed_image, image_latent, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if(adapters) {
            if(auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_clip_text_encoder->set_adapters(adapters);
            m_unet->set_adapters(adapters);
        }
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        using namespace numpy_utils;
        m_perf_metrics.clean_up();
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        const bool is_attentive = m_use_attentive_eraser && m_pipeline_type == PipelineType::INPAINTING;

        if (is_attentive) {
            OPENVINO_ASSERT(generation_config.attentive_eraser.has_value(),
                            "ImageGenerationConfig.attentive_eraser must be set in attentive eraser mode");
            OPENVINO_ASSERT(generation_config.guidance_scale == 1.0f,
                            "Attentive eraser mode requires guidance_scale == 1.0");
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt &&
                                generation_config.negative_prompt_2 == std::nullopt &&
                                generation_config.negative_prompt_3 == std::nullopt,
                            "Attentive eraser mode does not support negative prompts");
            OPENVINO_ASSERT(generation_config.strength > 0.0f && generation_config.strength <= 1.0f,
                            "Attentive eraser strength must be in (0, 1]");
            OPENVINO_ASSERT(generation_config.num_inference_steps > 0,
                            "Attentive eraser num_inference_steps must be positive");
            OPENVINO_ASSERT(generation_config.num_images_per_prompt == 1,
                            "Attentive eraser mode supports num_images_per_prompt == 1 only");
            OPENVINO_ASSERT(!generation_config.adapters.has_value(),
                            "Attentive eraser mode does not support LoRA adapters");
            OPENVINO_ASSERT(std::dynamic_pointer_cast<DDIMScheduler>(m_scheduler),
                            "Attentive Eraser mode requires a DDIM scheduler");
        }

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (generation_config.height < 0)
            compute_dim(generation_config.height, initial_image, 1 /* assume NHWC */);
        if (generation_config.width < 0)
            compute_dim(generation_config.width, initial_image, 2 /* assume NHWC */);

        if (is_attentive) {
            const int64_t model_image_size = static_cast<int64_t>(unet_config.sample_size * vae_scale_factor);
            OPENVINO_ASSERT(generation_config.height == model_image_size &&
                                generation_config.width == model_image_size,
                            "Attentive eraser height and width must match the UNet image size of ",
                            model_image_size);
        } else {
            check_inputs(generation_config, initial_image);
            set_lora_adapters(generation_config.adapters);
        }

        // use callback if defined
        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        // compute text encoders and set hidden states
        if (is_attentive) {
            compute_attentive_eraser_hidden_states(positive_prompt, generation_config);
        } else {
            compute_hidden_states(positive_prompt, generation_config);
        }

        // preparate initial / image latents
        ov::Tensor latent, processed_image, image_latent, noise;
        std::tie(latent, processed_image, image_latent, noise) = prepare_latents(initial_image, generation_config);

        // prepare mask latents
        ov::Tensor mask, masked_image_latent, latent_mask;
        if (m_pipeline_type == PipelineType::INPAINTING) {
            if (is_attentive) {
                ov::Tensor resized_mask = m_image_resizer->execute(mask_image,
                    generation_config.height, generation_config.width);
                const size_t configured_kernel = generation_config.attentive_eraser->mask_blur_kernel;
                ov::Tensor full_resolution_mask = preprocess_attentive_mask(
                    resized_mask,
                    configured_kernel == 0 ? attentive_eraser_mask_blur_kernel() : configured_kernel,
                    0.1f);
                latent_mask = max_pool_mask(full_resolution_mask, vae_scale_factor);
                m_unet->set_hidden_states("mask", full_resolution_mask);
                // start from noised image latent instead of pure noise
                image_latent.copy_to(latent);
                m_scheduler->add_noise(latent, noise, timesteps.front());
            } else {
                std::tie(mask, masked_image_latent) = prepare_mask_latents(mask_image, processed_image, generation_config, batch_size_multiplier);
            }
        }

        // prepare latents passed to models taking into account guidance scale (batch size multiplier)
        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;

        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg), denoised, noisy_residual_tensor(ov::element::f32, {}), latent_model_input;

        for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
            auto step_start = std::chrono::steady_clock::now();

            ov::Tensor noise_pred_tensor;
            if (is_attentive) {
                ov::Tensor current_step(ov::element::i64, {});
                *current_step.data<int64_t>() = static_cast<int64_t>(inference_step);
                ov::Tensor ss_steps_tensor(ov::element::i64, {});
                *ss_steps_tensor.data<int64_t>() = static_cast<int64_t>(generation_config.attentive_eraser->ss_steps);
                m_unet->set_hidden_states("cur_step", current_step);
                m_unet->set_hidden_states("ss_steps", ss_steps_tensor);

                ov::Tensor latent_pair = numpy_utils::repeat(latent, 2);
                ov::Tensor timestep(ov::element::i64, {});
                *timestep.data<int64_t>() = timesteps[inference_step];
                auto infer_start = std::chrono::steady_clock::now();
                ov::Tensor noise_pair = m_unet->infer(latent_pair, timestep);
                m_perf_metrics.raw_metrics.unet_inference_durations.emplace_back(
                    std::chrono::duration_cast<MicroSeconds>(std::chrono::steady_clock::now() - infer_start));

                noisy_residual_tensor = apply_attentive_removal_guidance(
                    noise_pair, generation_config.attentive_eraser->rm_guidance_scale);
            } else {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
                }

                m_scheduler->scale_model_input(latent_cfg, inference_step);

                ov::Tensor latent_model_input = is_inpainting_model() ? numpy_utils::concat(numpy_utils::concat(latent_cfg, mask, 1), masked_image_latent, 1) : latent_cfg;
                ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
                auto infer_start = std::chrono::steady_clock::now();
                noise_pred_tensor = m_unet->infer(latent_model_input, timestep);
                auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
                m_perf_metrics.raw_metrics.unet_inference_durations.emplace_back(MicroSeconds(infer_duration));

                ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
                noise_pred_shape[0] /= batch_size_multiplier;

                if (batch_size_multiplier > 1) {
                    noisy_residual_tensor.set_shape(noise_pred_shape);

                    float* noisy_residual = noisy_residual_tensor.data<float>();
                    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                    const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                    for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                        noisy_residual[i] = noise_pred_uncond[i] +
                            generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                    }
                } else {
                    noisy_residual_tensor = noise_pred_tensor;
                }
            }

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, generation_config.generator);
            latent = scheduler_step_result["latent"];

            if (is_attentive) {
                ov::Tensor initial_noised(image_latent.get_element_type(), image_latent.get_shape());
                image_latent.copy_to(initial_noised);
                if (inference_step + 1 < timesteps.size()) {
                    m_scheduler->add_noise(initial_noised, noise, timesteps[inference_step + 1]);
                }
                blend_attentive_latents(initial_noised, latent_mask, latent);
            } else if (m_pipeline_type == PipelineType::INPAINTING && !is_inpainting_model()) {
                blend_latents(image_latent, noise, mask, latent, inference_step);
            }

            // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
            const auto it = scheduler_step_result.find("denoised");
            denoised = it != scheduler_step_result.end() ? it->second : latent;

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), denoised) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

                auto image = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return image;
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }
        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }
        auto decode_start = std::chrono::steady_clock::now();
        auto image = decode(denoised);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        return m_vae->decode(latent);
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    static ov::Tensor apply_attentive_removal_guidance(const ov::Tensor& noise_pair, float scale) {
        OPENVINO_ASSERT(noise_pair.get_element_type() == ov::element::f32 &&
                            noise_pair.get_shape().size() == 4,
                        "Noise prediction must be a rank-4 f32 tensor");
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

    static void blend_attentive_latents(const ov::Tensor& initial_noised,
                                        const ov::Tensor& mask,
                                        ov::Tensor latents) {
        OPENVINO_ASSERT(initial_noised.get_element_type() == ov::element::f32 &&
                            mask.get_element_type() == ov::element::f32 &&
                            latents.get_element_type() == ov::element::f32,
                        "Attentive latent blending requires f32 tensors");
        OPENVINO_ASSERT(initial_noised.get_shape() == latents.get_shape(),
                        "Initial noised latent and latent shapes must match");
        const ov::Shape latent_shape = latents.get_shape();
        const ov::Shape mask_shape = mask.get_shape();
        OPENVINO_ASSERT(latent_shape.size() == 4 && mask_shape.size() == 4 &&
                            mask_shape[0] == latent_shape[0] && mask_shape[1] == 1 &&
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

    virtual size_t attentive_eraser_mask_blur_kernel() const {
        return 7;
    }

    virtual void compute_attentive_eraser_hidden_states(const std::string& positive_prompt,
                                                         const ImageGenerationConfig& generation_config) {
        const auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor hidden_states = m_clip_text_encoder->infer(positive_prompt, "", false);
        m_perf_metrics.encoder_inference_duration["text_encoder"] =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - infer_start).count();
        m_unet->set_hidden_states("encoder_hidden_states", numpy_utils::repeat(hidden_states, 2));
    }

    void initialize_attentive_eraser_generation_config() {
        if (!m_use_attentive_eraser) {
            return;
        }
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING,
                        "Attentive Eraser mode is available for inpainting pipelines only");
        OPENVINO_ASSERT(std::dynamic_pointer_cast<DDIMScheduler>(m_scheduler),
                        "Attentive Eraser mode requires a DDIM scheduler");
        apply_attentive_eraser_defaults(m_generation_config);
    }

    size_t get_config_in_channels() const override {
        assert(m_unet != nullptr);
        return m_unet->get_config().in_channels;
    }

    void compute_dim(int64_t & generation_config_value, ov::Tensor initial_image, int dim_idx) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& unet_config = m_unet->get_config();

        // in case of image to image generation_config_value is just ignored and computed based on initial image
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "Initial image is empty for image to image pipeline");
            ov::Shape shape = initial_image.get_shape();
            int64_t dim_val = shape[dim_idx];

            generation_config_value = dim_val - (dim_val % vae_scale_factor);
        }

        if (generation_config_value < 0)
            generation_config_value = unet_config.sample_size * vae_scale_factor;
    }

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_unet != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        // in case of image to image, the shape is computed based on initial image
        if (m_pipeline_type != PipelineType::IMAGE_2_IMAGE) {
            m_generation_config.height = unet_config.sample_size * vae_scale_factor;
            m_generation_config.width = unet_config.sample_size * vae_scale_factor;
        }

        if (class_name == "StableDiffusionPipeline" || class_name == "StableDiffusionImg2ImgPipeline" || class_name == "StableDiffusionInpaintPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else if (class_name == "LatentConsistencyModelPipeline" || class_name == "LatentConsistencyModelImg2ImgPipeline") {
            m_generation_config.guidance_scale = 8.5f;
            m_generation_config.num_inference_steps = 4;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_vae != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by ",
            vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        const bool is_classifier_free_guidance = m_unet->do_classifier_free_guidance(generation_config.guidance_scale);
        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "Latent Consistency Model" : "Stable Diffusion";

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
        if (is_lcm) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by ", pipeline_name);
        } else if (!is_classifier_free_guidance) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used when guidance scale <= 1.0");
        }
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ", pipeline_name);

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be within [0, 1] range");
        } else {
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModel> m_clip_text_encoder = nullptr;
    std::shared_ptr<UNet2DConditionModel> m_unet = nullptr;

    // Attentive eraser support
    bool m_use_attentive_eraser = false;
};

}  // namespace genai
}  // namespace ov
