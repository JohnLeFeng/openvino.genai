// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

inline constexpr const char* ATTENTIVE_ERASER_AAS_LAYERS = "ATTENTIVE_ERASER_AAS_LAYERS";

enum class AttentiveEraserUNetType {
    SD15,
    SD2,
    SDXL_BASE,
};

OPENVINO_GENAI_EXPORTS AttentiveEraserUNetType identify_attentive_eraser_unet(
    const std::shared_ptr<ov::Model>& model);

OPENVINO_GENAI_EXPORTS void apply_attentive_eraser_aas(const std::shared_ptr<ov::Model>& model,
                                                       const std::vector<size_t>& layer_indices);

}  // namespace genai
}  // namespace ov