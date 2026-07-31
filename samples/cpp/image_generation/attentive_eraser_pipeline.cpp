// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

#include <chrono>

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

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

    ov::Tensor image = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);

    ov::genai::InpaintingPipeline pipeline(
        models_path,
        device,
        ov::genai::inpainting_mode(ov::genai::InpaintingMode::ATTENTIVE_ERASER));

    ov::genai::ImageGenerationConfig config = pipeline.get_generation_config();
    config.strength = 1.0f;
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
