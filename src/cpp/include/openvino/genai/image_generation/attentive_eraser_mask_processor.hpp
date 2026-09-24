// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

struct AttentiveEraserMaskOutputs {
    ov::Tensor full_resolution;
    ov::Tensor pooled;
};

class AttentiveEraserMaskProcessor {
public:
    AttentiveEraserMaskProcessor(const std::string& device,
                                 size_t kernel_size,
                                 float threshold,
                                 bool gray_scale_source,
                                 size_t pooling_factor = 1);

    ov::Tensor execute(ov::Tensor mask);
    AttentiveEraserMaskOutputs execute_with_pooling(ov::Tensor mask);

private:
    void compile(std::shared_ptr<ov::Model> model, const std::string& device);
    void validate(const ov::Tensor& mask) const;

    ov::InferRequest m_request;
    size_t m_padding_radius;
    size_t m_pooling_factor;
    bool m_gray_scale_source;
};

}  // namespace genai
}  // namespace ov
