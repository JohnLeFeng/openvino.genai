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

struct OPENVINO_GENAI_EXPORTS SDXLAttentiveEraserConfig {
    float rm_guidance_scale = 9.0f;
    size_t ss_steps = 9;
    float strength = 0.8f;
    size_t num_inference_steps = 50;
    size_t rng_seed = 123;
    std::shared_ptr<Generator> generator = nullptr;
    std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;

    void validate() const;
};

class OPENVINO_GENAI_EXPORTS SDXLAttentiveEraser {
public:
    explicit SDXLAttentiveEraser(const std::filesystem::path& models_path);

    SDXLAttentiveEraser(const std::filesystem::path& models_path,
                        const std::string& device,
                        const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    SDXLAttentiveEraser(const std::filesystem::path& models_path,
                        const std::string& device,
                        Properties&&... properties)
        : SDXLAttentiveEraser(models_path,
                              device,
                              ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~SDXLAttentiveEraser();

    SDXLAttentiveEraser(SDXLAttentiveEraser&&) noexcept;
    SDXLAttentiveEraser& operator=(SDXLAttentiveEraser&&) noexcept;
    SDXLAttentiveEraser(const SDXLAttentiveEraser&) = delete;
    SDXLAttentiveEraser& operator=(const SDXLAttentiveEraser&) = delete;

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    SDXLAttentiveEraserConfig get_generation_config() const;
    void set_generation_config(const SDXLAttentiveEraserConfig& generation_config);

    ov::Tensor generate(ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const SDXLAttentiveEraserConfig& generation_config);

    ov::Tensor generate(ov::Tensor initial_image, ov::Tensor mask_image);
    ov::Tensor decode(const ov::Tensor& latent);
    ImageGenerationPerfMetrics get_performance_metrics();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace genai
}  // namespace ov