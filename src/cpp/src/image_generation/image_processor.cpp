// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/select.hpp"

#include "utils.hpp" // for utils::singleton_core

namespace ov {
namespace genai {

namespace {

std::shared_ptr<ov::Model> create_empty_model(ov::element::Type type = ov::element::f32) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape::dynamic(4));
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
}

} // namespace

IImageProcessor::IImageProcessor(const std::string& device) :
    m_device(device) {
}

ov::Tensor IImageProcessor::execute(ov::Tensor image) {
    OPENVINO_ASSERT(m_request, "ImageProcessor model must be compiled first. Cannot infer non-compiled model");
    m_request.set_input_tensor(image);
    m_request.infer();
    return m_request.get_output_tensor();
}

void IImageProcessor::compile(std::shared_ptr<ov::Model> model) {
    m_request = utils::singleton_core().compile_model(model, m_device).create_infer_request();
}

ImageProcessor::ImageProcessor(const std::string& device, bool do_normalize, bool do_binarize, bool gray_scale_source) :
    IImageProcessor(device) {
    auto image_processor_model = create_empty_model();
    merge_image_preprocessing(image_processor_model, do_normalize, do_binarize, gray_scale_source);

    compile(std::move(image_processor_model));
}

void ImageProcessor::merge_image_preprocessing(std::shared_ptr<ov::Model> model, bool do_normalize, bool do_binarize, bool gray_scale_source) {
    OPENVINO_ASSERT(do_normalize ^ do_binarize, "Both binarize and normalize are not supported");

    // https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L90-L110
    ov::preprocess::PrePostProcessor ppp(model);

    ov::preprocess::ColorFormat source_color_format = gray_scale_source ? ov::preprocess::ColorFormat::GRAY : ov::preprocess::ColorFormat::RGB;

    ppp.input().tensor()
        .set_layout("NHWC")
        .set_element_type(ov::element::u8)
        .set_color_format(source_color_format);
    ppp.input().model()
        .set_layout("NCHW");

    if (do_normalize) {
        ppp.input().preprocess()
            .convert_layout()
            .convert_element_type(ov::element::f32)
            // this is less accurate that in VaeImageProcessor::normalize
            .scale(255.0 / 2.0)
            .mean(1.0f);
    } else if (do_binarize) {
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::GRAY)
            .scale(255.0f)
            .custom([](const ov::Output<ov::Node>& port) {
                auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
                auto constant_1_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 1.0f);
                auto constant_0_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.0f);
                auto mask_bool = std::make_shared<ov::op::v1::GreaterEqual>(port, constant_0_5);
                auto mask_float = std::make_shared<ov::op::v1::Select>(mask_bool, constant_1_0, constant_0_0);
                return mask_float;
            });
    }

    ppp.build();
}

ImageResizer::ImageResizer(const std::string& device, ov::element::Type type, ov::Layout layout, ov::op::v11::Interpolate::InterpolateMode interpolation_mode) {
    auto image_parameter = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape::dynamic(4));
    image_parameter->get_output_tensor(0).add_names({"image"});

    auto target_spatial_shape = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});
    target_spatial_shape->get_output_tensor(0).add_names({"target_spatial_shape"});

    ov::PartialShape pshape = ov::PartialShape::dynamic(4);
    const auto height_idx = static_cast<int64_t>(get_and_check_height_idx(layout, pshape));
    const auto width_idx = static_cast<int64_t>(get_and_check_width_idx(layout, pshape));

    // In future consider replacing this to set of new OV operations like `getDimByName(node, "H")`
    // This is to allow specifying layout on 'evaluation' stage
    const auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {height_idx, width_idx});

    op::util::InterpolateBase::InterpolateAttrs attrs(interpolation_mode,
                                                        op::util::InterpolateBase::ShapeCalcMode::SIZES,
                                                        {0, 0},
                                                        {0, 0});

    attrs.coordinate_transformation_mode = op::util::InterpolateBase::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = op::util::InterpolateBase::NearestMode::FLOOR;
    if (attrs.mode != op::util::InterpolateBase::InterpolateMode::NEAREST) {
        attrs.coordinate_transformation_mode = op::util::InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    }

    const auto interp = std::make_shared<op::v11::Interpolate>(image_parameter, target_spatial_shape, axes, attrs);

    auto result = std::make_shared<ov::op::v0::Result>(interp);
    auto resize_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{image_parameter, target_spatial_shape});

    m_request = utils::singleton_core().compile_model(resize_model, device).create_infer_request();
}

ov::Tensor ImageResizer::execute(ov::Tensor image, int64_t dst_height, int64_t dst_width) {
    OPENVINO_ASSERT(m_request, "ImageResizer model must be compiled first. Cannot infer non-compiled model");
    ov::Tensor target_spatial_tensor(ov::element::i64, ov::Shape{2});
    target_spatial_tensor.data<int64_t>()[0] = dst_height;
    target_spatial_tensor.data<int64_t>()[1] = dst_width;

    m_request.set_tensor("image", image);
    m_request.set_tensor("target_spatial_shape", target_spatial_tensor);
    m_request.infer();

    return m_request.get_output_tensor();
}

size_t ImageResizer::get_and_check_width_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_width(layout), "Layout ", layout.to_string(), " doesn't have `width` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape width index for shape with dynamic rank");
    auto idx = ov::layout::width_idx(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Width dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

size_t ImageResizer::get_and_check_height_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_height(layout), "Layout ", layout.to_string(), " doesn't have `height` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape height index for shape with dynamic rank");
    auto idx = ov::layout::height_idx(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Height dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

namespace {

void validate_f32_nchw(const ov::Tensor& tensor, const char* name) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32, name, " must have f32 element type");
    OPENVINO_ASSERT(tensor.get_shape().size() == 4, name, " must be a rank-4 NCHW tensor");
}

size_t reflect_index(int64_t index, size_t size) {
    OPENVINO_ASSERT(size > 1, "Reflection padding requires a spatial dimension greater than one");
    while (index < 0 || index >= static_cast<int64_t>(size)) {
        index = index < 0 ? -index : 2 * static_cast<int64_t>(size) - index - 2;
    }
    return static_cast<size_t>(index);
}

}  // namespace

ov::Tensor gaussian_blur_and_binarize_mask(const ov::Tensor& gray_mask,
                                            size_t kernel_size,
                                            float threshold) {
    validate_f32_nchw(gray_mask, "Mask");
    const ov::Shape shape = gray_mask.get_shape();
    OPENVINO_ASSERT(shape[1] == 1, "Mask must have one channel");
    OPENVINO_ASSERT(kernel_size > 0 && kernel_size % 2 == 1,
                    "Gaussian kernel size must be positive and odd");
    OPENVINO_ASSERT(threshold >= 0.0f && threshold <= 1.0f, "Mask threshold must be in [0, 1]");

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

    ov::Tensor horizontal(ov::element::f32, shape);
    ov::Tensor result(ov::element::f32, shape);
    const float* source = gray_mask.data<const float>();
    float* horizontal_data = horizontal.data<float>();
    float* result_data = result.data<float>();
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const size_t batch_offset = batch * height * width;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_x = reflect_index(static_cast<int64_t>(x) +
                                                              static_cast<int64_t>(kernel_index) - radius,
                                                          width);
                    value += source[batch_offset + y * width + source_x] * kernel[kernel_index];
                }
                horizontal_data[batch_offset + y * width + x] = value;
            }
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float value = 0.0f;
                for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const size_t source_y = reflect_index(static_cast<int64_t>(y) +
                                                              static_cast<int64_t>(kernel_index) - radius,
                                                          height);
                    value += horizontal_data[batch_offset + source_y * width + x] * kernel[kernel_index];
                }
                result_data[batch_offset + y * width + x] = value < threshold ? 0.0f : 1.0f;
            }
        }
    }
    return result;
}

ov::Tensor max_pool_mask(const ov::Tensor& mask, size_t factor) {
    validate_f32_nchw(mask, "Mask");
    const ov::Shape shape = mask.get_shape();
    OPENVINO_ASSERT(shape[1] == 1, "Mask must have one channel");
    OPENVINO_ASSERT(factor > 0 && shape[2] % factor == 0 && shape[3] % factor == 0,
                    "Mask dimensions must be divisible by the pooling factor");

    const ov::Shape output_shape{shape[0], 1, shape[2] / factor, shape[3] / factor};
    ov::Tensor result(ov::element::f32, output_shape);
    const float* source = mask.data<const float>();
    float* destination = result.data<float>();
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        for (size_t output_y = 0; output_y < output_shape[2]; ++output_y) {
            for (size_t output_x = 0; output_x < output_shape[3]; ++output_x) {
                float maximum = 0.0f;
                for (size_t y = 0; y < factor; ++y) {
                    for (size_t x = 0; x < factor; ++x) {
                        const size_t source_index = batch * shape[2] * shape[3] +
                                                    (output_y * factor + y) * shape[3] +
                                                    output_x * factor + x;
                        maximum = std::max(maximum, source[source_index]);
                    }
                }
                destination[batch * output_shape[2] * output_shape[3] +
                            output_y * output_shape[3] + output_x] = maximum;
            }
        }
    }
    return result;
}

ov::Tensor preprocess_attentive_mask(const ov::Tensor& mask,
                                      size_t kernel_size,
                                      float threshold) {
    const ov::Shape& shape = mask.get_shape();
    OPENVINO_ASSERT(mask.get_element_type() == ov::element::u8, "Mask must have u8 element type");
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1, "Mask must be rank-4 NHWC with batch 1");
    const size_t channels = shape[3];
    OPENVINO_ASSERT(channels == 1 || channels == 3, "Mask must have 1 or 3 channels");

    ov::Tensor gray_mask(ov::element::f32, {1, 1, shape[1], shape[2]});
    const uint8_t* source = mask.data<const uint8_t>();
    float* destination = gray_mask.data<float>();
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[2]; ++x) {
            const size_t source_index = (y * shape[2] + x) * channels;
            destination[y * shape[2] + x] = channels == 1
                ? static_cast<float>(source[source_index]) / 255.0f
                : (0.299f * source[source_index] + 0.587f * source[source_index + 1] +
                   0.114f * source[source_index + 2]) / 255.0f;
        }
    }
    return gaussian_blur_and_binarize_mask(gray_mask, kernel_size, threshold);
}

} // namespace genai
} // namespace ov
