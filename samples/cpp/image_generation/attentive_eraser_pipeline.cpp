// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sd_attentive_eraser.hpp"
#include "openvino/genai/image_generation/sdxl_attentive_eraser.hpp"

#include <chrono>

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 5 && argc <= 7,
                    "Usage: ",
                    argv[0],
                    " <SD15|SD2|SDXL> <MODEL_DIR> <IMAGE> <MASK_IMAGE> [DEVICE] [SEED]");

    const std::string model_type = argv[1];
    const std::string models_path = argv[2];
    const std::string image_path = argv[3];
    const std::string mask_image_path = argv[4];
    const std::string device = argc >= 6 ? argv[5] : "CPU";
    OPENVINO_ASSERT(model_type == "SD15" || model_type == "SD2" || model_type == "SDXL",
                    "Unsupported model type '",
                    model_type,
                    "'. Expected SD15, SD2, or SDXL");
    const size_t seed = argc == 7 ? std::stoull(argv[6]) : (model_type == "SDXL" ? 123 : 0);

    ov::Tensor image = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);

    ov::Tensor generated_image;
    std::chrono::duration<double> generation_duration;
    if (model_type == "SD15") {
        ov::genai::SDAttentiveEraserConfig config;
        config.callback = progress_bar;
        config.rng_seed = seed;
        ov::genai::SD15AttentiveEraser pipeline(models_path, device);
        const auto generation_start = std::chrono::steady_clock::now();
        generated_image = pipeline.generate(image, mask_image, config);
        generation_duration = std::chrono::steady_clock::now() - generation_start;
        imwrite("removed_image_SD15.bmp", generated_image, true);
    } else if (model_type == "SD2") {
        ov::genai::SDAttentiveEraserConfig config;
        config.callback = progress_bar;
        config.rng_seed = seed;
        ov::genai::SD2AttentiveEraser pipeline(models_path, device);
        const auto generation_start = std::chrono::steady_clock::now();
        generated_image = pipeline.generate(image, mask_image, config);
        generation_duration = std::chrono::steady_clock::now() - generation_start;
        imwrite("removed_image_SD2.bmp", generated_image, true);
    } else if (model_type == "SDXL") {
        ov::genai::SDXLAttentiveEraserConfig config;
        config.callback = progress_bar;
        config.rng_seed = seed;
        ov::genai::SDXLAttentiveEraser pipeline(models_path, device);
        const auto generation_start = std::chrono::steady_clock::now();
        generated_image = pipeline.generate(image, mask_image, config);
        generation_duration = std::chrono::steady_clock::now() - generation_start;
        imwrite("removed_image_SDXL.bmp", generated_image, true);
    }
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
