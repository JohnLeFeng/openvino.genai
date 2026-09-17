// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_generation/image_processor.hpp"

namespace ov {
namespace genai {

class AttentiveEraserMaskProcessor : public IImageProcessor {
public:
    AttentiveEraserMaskProcessor(const std::string& device,
                                 size_t kernel_size,
                                 float threshold,
                                 bool gray_scale_source);

    ov::Tensor execute(ov::Tensor mask) override;

private:
    size_t m_padding_radius;
    bool m_gray_scale_source;
};

}  // namespace genai
}  // namespace ov
