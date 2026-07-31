// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

namespace {

ov::Tensor resize_nearest(const ov::Tensor& input, size_t target_height, size_t target_width) {
    const ov::Shape& shape = input.get_shape();
    OPENVINO_ASSERT(input.get_element_type() == ov::element::u8 && shape.size() == 4,
                    "Input must be a rank-4 NHWC u8 tensor");
    if (shape[1] == target_height && shape[2] == target_width) {
        return input;
    }

    const size_t batch_size = shape[0];
    const size_t source_height = shape[1];
    const size_t source_width = shape[2];
    const size_t channels = shape[3];
    ov::Tensor result(ov::element::u8, {batch_size, target_height, target_width, channels});
    const uint8_t* source = input.data<const uint8_t>();
    uint8_t* destination = result.data<uint8_t>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t y = 0; y < target_height; ++y) {
            const size_t source_y = y * source_height / target_height;
            for (size_t x = 0; x < target_width; ++x) {
                const size_t source_x = x * source_width / target_width;
                const size_t source_index = ((batch * source_height + source_y) * source_width + source_x) * channels;
                const size_t destination_index = ((batch * target_height + y) * target_width + x) * channels;
                std::copy_n(source + source_index, channels, destination + destination_index);
            }
        }
    }
    return result;
}

}  // namespace

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 4 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> <IMAGE> <MASK_IMAGE> [DEVICE] [SEED]");

    const std::string models_path = argv[1];
    const std::string image_path = argv[2];
    const std::string mask_image_path = argv[3];
    const std::string device = argc >= 5 ? argv[4] : "CPU";
    const size_t seed = argc == 6 ? std::stoull(argv[5]) : 123;

    ov::genai::InpaintingPipeline pipeline(
        models_path,
        device,
        ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));

    ov::genai::ImageGenerationConfig config = pipeline.get_generation_config();
    const size_t target_height = static_cast<size_t>(config.height);
    const size_t target_width = static_cast<size_t>(config.width);
    ov::Tensor image = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);
    const bool needs_resize = image.get_shape()[1] != target_height || image.get_shape()[2] != target_width ||
                              mask_image.get_shape()[1] != target_height || mask_image.get_shape()[2] != target_width;
    if (needs_resize) {
        std::cout << "Resizing image and mask to " << target_width << 'x' << target_height << '\n';
    }
    image = resize_nearest(image, target_height, target_width);
    mask_image = resize_nearest(mask_image, target_height, target_width);

    config.strength = 0.8f;
    config.num_inference_steps = 50;
    config.rng_seed = seed;
    config.attentive_eraser->rm_guidance_scale = 9.0f;
    config.attentive_eraser->ss_steps = 9;
    pipeline.set_generation_config(config);

    const auto generation_start = std::chrono::steady_clock::now();
    ov::Tensor generated_image = pipeline.generate(
        "",
        image,
        mask_image,
        ov::genai::callback(progress_bar));
    const std::chrono::duration<double> generation_duration =
        std::chrono::steady_clock::now() - generation_start;

    imwrite("object_removed_image.bmp", generated_image, true);
    std::cout << "Generation with seed " << seed << " completed in " << generation_duration.count() << " s\n";

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
