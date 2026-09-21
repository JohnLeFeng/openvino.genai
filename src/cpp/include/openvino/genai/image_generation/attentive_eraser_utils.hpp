// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

OPENVINO_GENAI_EXPORTS bool is_attentive_eraser_aas_active(size_t inference_step,
                                                           size_t start_step,
                                                           float strength,
                                                           size_t num_inference_steps);

OPENVINO_GENAI_EXPORTS bool is_attentive_eraser_ss_active(size_t inference_step, size_t ss_steps);

}  // namespace genai
}  // namespace ov