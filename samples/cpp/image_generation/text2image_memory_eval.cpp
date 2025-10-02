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

#include <thread>

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


class t2iPipeline
{
public:
    t2iPipeline() {
    }

    ~t2iPipeline() {
    }

    void load_model(const std::string models_path, const std::string device) {
        ov::AnyMap properties = {
            {"ENABLE_MMAP", false}
        };
        if (device == "GPU") {
            properties.insert({ov::cache_dir("cache")});
        }
        if (!t2i_pipeline) {
            std::cout << "Wrapper - Initial t2i pipeline.\n";
            t2i_pipeline = std::make_unique<ov::genai::Text2ImagePipeline>(models_path, device, properties);
        }
    }

    // ov::Tensor generate(const std::string prompt) {
    //     ov::Tensor image = t2i_pipeline->generate(prompt,
    //         ov::genai::width(512),
    //         ov::genai::height(512),
    //         ov::genai::num_inference_steps(20),
    //         ov::genai::num_images_per_prompt(1),
    //         ov::genai::callback(progress_bar));
        
    //     return image;
    // }

    void generate(const std::string prompt) {
        ov::Tensor image = t2i_pipeline->generate(prompt,
            ov::genai::width(512),
            ov::genai::height(512),
            ov::genai::num_inference_steps(20),
            ov::genai::num_images_per_prompt(1),
            ov::genai::callback(progress_bar));
    }

    void release() {
        t2i_pipeline.reset();
        t2i_pipeline.release();
        if (!t2i_pipeline) {
            std::cout << "Wrapper - t2i pipeline is nullptr.\n";
        }
    }

private:
   std::unique_ptr<ov::genai::Text2ImagePipeline> t2i_pipeline;
};


int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' \n\t\t<DISABLE_PRIMITIVE_CACHE:0;ENABLE_PRIMITIVE_CACHE, but set CACHE CAPACITY as 0:1;default:others> \n\t\t[<DEVICE>]");

    const std::string models_path = argv[1], prompt = argv[2];
    const int ONEDNN_CACHE = std::stoi(argv[3]);

    switch (ONEDNN_CACHE) {
        case 0:
            std::cout << "Disable OneDNN PRIMITIVE CACHE." << std::endl;
            if (_putenv("ONEDNN_ENABLE_PRIMITIVE_CACHE=OFF") != 0) {
                throw std::runtime_error("Failed to set environment variable.");
            }
            if (_putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=0") != 0) {
                throw std::runtime_error("Failed to set environment variable.");
            }
            break;
        case 1:
            std::cout << "Enable OneDNN PRIMITIVE CACHE, but set CACHE CAPACITY as 0." << std::endl;
            if (_putenv("ONEDNN_ENABLE_PRIMITIVE_CACHE=ON") != 0) {
                throw std::runtime_error("Failed to set environment variable.");
            }
            if (_putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=0") != 0) {
                throw std::runtime_error("Failed to set environment variable.");
            }
            break;
        default:
            std::cout << "Enable OneDNN PRIMITIVE CACHE." << std::endl;
    }
        
    const std::string device = (argc == 5) ? argv[4] : "GPU";

    // 
    // Original sample
    // 
    // printMemoryUsage("Initial Text2Image Pipeline ...");
    // ov::genai::Text2ImagePipeline pipe(models_path, device);
    // printMemoryUsage("Done");

    // printMemoryUsage("Run Text2Image generation ...");
    // ov::Tensor image = pipe.generate(prompt,
    //     ov::genai::width(512),
    //     ov::genai::height(512),
    //     ov::genai::num_inference_steps(20),
    //     ov::genai::num_images_per_prompt(1),
    //     ov::genai::callback(progress_bar));
    // printMemoryUsage("Done");

    // printMemoryUsage("Run save image ...");
    // // writes `num_images_per_prompt` images by pattern name
    // imwrite("image_%d.bmp", image, true);
    // printMemoryUsage("Done");

    // 
    // Use unique pointer 
    // 
    printMemoryUsage("Initial Text2Image Pipeline Wrapper ...");
    t2iPipeline t2iPipe;
    printMemoryUsage("Done");

    printMemoryUsage("Load Text2Image Pipeline ...");
    t2iPipe.load_model(models_path, device);
    printMemoryUsage("Done");

    printMemoryUsage("Generate Text2Image Pipeline ...");
    t2iPipe.generate(prompt);
    printMemoryUsage("Done");

    printMemoryUsage("Release Text2Image Pipeline ...");
    t2iPipe.release();
    printMemoryUsage("Done");


    for (int i = 0; i < 12; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::string delay_msg = "Release Text2Image Pipeline after " + std::to_string(i * 5 + 5) + " seconds";
        printMemoryUsage(delay_msg);
    }

    std::cout << std::endl << "Re-initial ..." << std::endl << std::endl;

    printMemoryUsage("Load Text2Image Pipeline ...");
    t2iPipe.load_model(models_path, device);
    printMemoryUsage("Done");

    printMemoryUsage("Generate Text2Image Pipeline ...");
    t2iPipe.generate(prompt);
    printMemoryUsage("Done");

    printMemoryUsage("Release Text2Image Pipeline ...");
    t2iPipe.release();
    printMemoryUsage("Done");

    for (int i = 0; i < 12; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::string delay_msg = "Release Text2Image Pipeline after " + std::to_string(i * 5 + 5) + " seconds";
        printMemoryUsage(delay_msg);
    }

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
