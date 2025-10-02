// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <psapi.h>

size_t getCurrentRSS() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

void printMemoryUsage(const std::string& stage) {
    size_t rss = getCurrentRSS();
    if (rss > 0) {
        // For better readability, convert bytes to MB
        double rss_mb = static_cast<double>(rss) / (1024 * 1024);
        std::cout << stage << " - Current RSS: " << rss << " bytes (" << rss_mb << " MB)" << std::endl;
    } else {
        std::cout << stage << " - Failed to get RSS." << std::endl;
    }
}

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<DEVICE>]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = (argc == 4) ? argv[3] : "GPU";

    printMemoryUsage("Initial Text2Image Pipeline");
    ov::genai::Text2ImagePipeline pipe(models_path, device);

    printMemoryUsage("Run Text2Image generation");
    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar));

    printMemoryUsage("Run save image");
    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", image, true);

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
