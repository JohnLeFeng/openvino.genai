// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sdxl_attentive_eraser.hpp"

#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 4 || argc == 5,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> <IMAGE> <MASK_IMAGE> [DEVICE]");

    const std::string models_path = argv[1];
    const std::string image_path = argv[2];
    const std::string mask_image_path = argv[3];
    const std::string device = argc == 5 ? argv[4] : "CPU";

    ov::Tensor image = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);

    ov::genai::SDXLAttentiveEraserConfig config;
    config.callback = progress_bar;

    ov::genai::SDXLAttentiveEraser pipeline(models_path, device);
    ov::Tensor generated_image = pipeline.generate(image, mask_image, config);
    imwrite("removed_image.bmp", generated_image, true);

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