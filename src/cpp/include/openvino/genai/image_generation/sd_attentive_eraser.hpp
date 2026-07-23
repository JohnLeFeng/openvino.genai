// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

struct OPENVINO_GENAI_EXPORTS SDAttentiveEraserConfig {
    float rm_guidance_scale = 9.0f;
    size_t ss_steps = 9;
    float strength = 0.8f;
    size_t num_inference_steps = 50;
    size_t rng_seed = 123;
    std::shared_ptr<Generator> generator = nullptr;
    std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;

    void validate() const;
};

namespace detail {
class SDAttentiveEraserImpl;
}

class OPENVINO_GENAI_EXPORTS SD15AttentiveEraser {
public:
    explicit SD15AttentiveEraser(const std::filesystem::path& models_path);
    SD15AttentiveEraser(const std::filesystem::path& models_path,
                        const std::string& device,
                        const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    SD15AttentiveEraser(const std::filesystem::path& models_path,
                        const std::string& device,
                        Properties&&... properties)
        : SD15AttentiveEraser(models_path,
                              device,
                              ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~SD15AttentiveEraser();
    SD15AttentiveEraser(SD15AttentiveEraser&&) noexcept;
    SD15AttentiveEraser& operator=(SD15AttentiveEraser&&) noexcept;
    SD15AttentiveEraser(const SD15AttentiveEraser&) = delete;
    SD15AttentiveEraser& operator=(const SD15AttentiveEraser&) = delete;

    void compile(const std::string& device, const ov::AnyMap& properties = {});
    SDAttentiveEraserConfig get_generation_config() const;
    void set_generation_config(const SDAttentiveEraserConfig& generation_config);
    ov::Tensor generate(ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const SDAttentiveEraserConfig& generation_config);
    ov::Tensor generate(ov::Tensor initial_image, ov::Tensor mask_image);
    ov::Tensor decode(const ov::Tensor& latent);
    ImageGenerationPerfMetrics get_performance_metrics();

private:
    std::unique_ptr<detail::SDAttentiveEraserImpl> m_impl;
};

class OPENVINO_GENAI_EXPORTS SD2AttentiveEraser {
public:
    explicit SD2AttentiveEraser(const std::filesystem::path& models_path);
    SD2AttentiveEraser(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    SD2AttentiveEraser(const std::filesystem::path& models_path,
                       const std::string& device,
                       Properties&&... properties)
        : SD2AttentiveEraser(models_path,
                             device,
                             ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~SD2AttentiveEraser();
    SD2AttentiveEraser(SD2AttentiveEraser&&) noexcept;
    SD2AttentiveEraser& operator=(SD2AttentiveEraser&&) noexcept;
    SD2AttentiveEraser(const SD2AttentiveEraser&) = delete;
    SD2AttentiveEraser& operator=(const SD2AttentiveEraser&) = delete;

    void compile(const std::string& device, const ov::AnyMap& properties = {});
    SDAttentiveEraserConfig get_generation_config() const;
    void set_generation_config(const SDAttentiveEraserConfig& generation_config);
    ov::Tensor generate(ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const SDAttentiveEraserConfig& generation_config);
    ov::Tensor generate(ov::Tensor initial_image, ov::Tensor mask_image);
    ov::Tensor decode(const ov::Tensor& latent);
    ImageGenerationPerfMetrics get_performance_metrics();

private:
    std::unique_ptr<detail::SDAttentiveEraserImpl> m_impl;
};

}  // namespace genai
}  // namespace ov