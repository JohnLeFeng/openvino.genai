// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/attentive_eraser_mask_processor.hpp"

#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace genai {

namespace {

std::shared_ptr<ov::Model> create_attentive_mask_model(size_t kernel_size,
                                                       float threshold,
                                                       bool gray_scale_source) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});

    ov::preprocess::PrePostProcessor preprocessing(model);
    const ov::preprocess::ColorFormat source_color_format = gray_scale_source
        ? ov::preprocess::ColorFormat::GRAY
        : ov::preprocess::ColorFormat::RGB;
    preprocessing.input().tensor()
        .set_layout("NHWC")
        .set_element_type(ov::element::u8)
        .set_color_format(source_color_format);
    preprocessing.input().model().set_layout("NCHW");
    preprocessing.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::GRAY)
        .scale(255.0f)
        .convert_layout()
        .custom([kernel_size, threshold](const ov::Output<ov::Node>& port) {
            const float sigma = 0.3f * ((static_cast<float>(kernel_size) - 1.0f) * 0.5f - 1.0f) + 0.8f;
            const int64_t radius = static_cast<int64_t>(kernel_size / 2);
            std::vector<float> kernel(kernel_size);
            for (size_t index = 0; index < kernel_size; ++index) {
                const float distance = static_cast<float>(static_cast<int64_t>(index) - radius);
                kernel[index] = std::exp(-(distance * distance) / (2.0f * sigma * sigma));
            }
            const float kernel_sum = std::accumulate(kernel.begin(), kernel.end(), 0.0f);
            for (float& value : kernel) {
                value /= kernel_sum;
            }

            const auto horizontal_kernel = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape{1, 1, 1, kernel_size}, kernel);
            const auto vertical_kernel = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape{1, 1, kernel_size, 1}, kernel);
            const auto horizontal_padding = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, radius});
            const auto vertical_padding = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, radius, 0});

            ov::Output<ov::Node> processed = std::make_shared<ov::op::v1::Pad>(
                port, horizontal_padding, horizontal_padding, ov::op::PadMode::REFLECT);
            processed = std::make_shared<ov::op::v1::Convolution>(
                processed, horizontal_kernel, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
                ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
            processed = std::make_shared<ov::op::v1::Pad>(
                processed, vertical_padding, vertical_padding, ov::op::PadMode::REFLECT);
            processed = std::make_shared<ov::op::v1::Convolution>(
                processed, vertical_kernel, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
                ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

            const auto threshold_node = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1}, threshold);
            const auto one = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 1.0f);
            const auto zero = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.0f);
            const auto binary_mask = std::make_shared<ov::op::v1::GreaterEqual>(processed, threshold_node);
            return std::make_shared<ov::op::v1::Select>(binary_mask, one, zero)->output(0);
        });
    preprocessing.build();
    return model;
}

}  // namespace

AttentiveEraserMaskProcessor::AttentiveEraserMaskProcessor(const std::string& device,
                                                           size_t kernel_size,
                                                           float threshold,
                                                           bool gray_scale_source) :
    IImageProcessor(device),
    m_padding_radius(kernel_size / 2),
    m_gray_scale_source(gray_scale_source) {
    OPENVINO_ASSERT(kernel_size > 0 && kernel_size % 2 == 1,
                    "Gaussian kernel size must be positive and odd");
    OPENVINO_ASSERT(threshold >= 0.0f && threshold <= 1.0f, "Mask threshold must be in [0, 1]");
    compile(create_attentive_mask_model(kernel_size, threshold, gray_scale_source));
}

ov::Tensor AttentiveEraserMaskProcessor::execute(ov::Tensor mask) {
    const ov::Shape& shape = mask.get_shape();
    OPENVINO_ASSERT(mask.get_element_type() == ov::element::u8, "Mask must have u8 element type");
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1, "Mask must be rank-4 NHWC with batch 1");
    const size_t expected_channels = m_gray_scale_source ? 1 : 3;
    OPENVINO_ASSERT(shape[3] == expected_channels, "Mask must have ", expected_channels, " channels");
    OPENVINO_ASSERT(m_padding_radius < shape[1] && m_padding_radius < shape[2],
                    "Gaussian reflection padding radius ", m_padding_radius,
                    " must be smaller than mask dimensions ", shape[1], "x", shape[2]);
    return IImageProcessor::execute(std::move(mask));
}

}  // namespace genai
}  // namespace ov
