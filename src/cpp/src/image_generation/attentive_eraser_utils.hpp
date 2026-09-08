// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor);

ov::Tensor preprocess_attentive_mask(const ov::Tensor& mask,
                                     size_t kernel_size,
                                     float threshold);

}  // namespace genai
}  // namespace ov
