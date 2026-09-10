// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace genai {

inline constexpr const char* ATTENTIVE_ERASER_AAS_LAYERS = "ATTENTIVE_ERASER_AAS_LAYERS";

enum class AttentiveEraserUNetType {
    SD15,
    SDXL_BASE,
};

AttentiveEraserUNetType identify_attentive_eraser_unet(const std::shared_ptr<ov::Model>& model);

void apply_attentive_eraser_aas(const std::shared_ptr<ov::Model>& model,
                                const std::vector<size_t>& layer_indices);

}  // namespace genai
}  // namespace ov