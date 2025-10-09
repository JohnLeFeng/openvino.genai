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

#include <tlhelp32.h>

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


void PrintLoadedDlls() {
    // Get the process ID of the current process.
    DWORD currentProcessId = GetCurrentProcessId();

    // 1. Create a snapshot of all modules in the current process.
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, currentProcessId);

    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: CreateToolhelp32Snapshot failed. Code: " << GetLastError() << std::endl;
        return;
    }

    // 2. Initialize the MODULEENTRY32 structure.
    // You must set the dwSize member before calling Module32First.
    MODULEENTRY32 moduleEntry;
    moduleEntry.dwSize = sizeof(MODULEENTRY32);

    // 3. Get information about the first module (the .exe itself).
    if (Module32First(hSnapshot, &moduleEntry)) {
        std::cout << "--- Loaded Modules ---" << std::endl;
        do {
            // szExePath contains the full path to the module.
            // On Windows, paths can contain backslashes, so we print it directly.
            // Using std::wstring for wider compatibility with Windows paths.
            std::wcout << L"  " << moduleEntry.szExePath << std::endl;

        } while (Module32Next(hSnapshot, &moduleEntry)); // 4. Loop through the rest of the modules.
    } else {
        std::cerr << "Error: Module32First failed. Code: " << GetLastError() << std::endl;
    }

    // 5. Clean up and close the snapshot handle.
    CloseHandle(hSnapshot);
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
            {"ENABLE_MMAP", true}
            // {"ENABLE_MMAP", false}
        };
        if (device == "GPU") {
            properties.insert({ov::cache_dir("cache")});
        }
        if (!t2i_pipeline) {
            // std::cout << "Wrapper - Initial t2i pipeline.\n";
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
            ov::genai::num_images_per_prompt(1)
            // ov::genai::num_images_per_prompt(1),
            // ov::genai::callback(progress_bar)
        );
    }

    void release() {
        t2i_pipeline.reset();
        t2i_pipeline.release();
        // if (!t2i_pipeline) {
        //     std::cout << "Wrapper - t2i pipeline is nullptr.\n";
        // }
    }

private:
   std::unique_ptr<ov::genai::Text2ImagePipeline> t2i_pipeline;
};


int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<DEVICE>]");

    const std::string models_path = argv[1], prompt = argv[2];
        
    const std::string device = (argc == 4) ? argv[3] : "GPU";

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
    for (int i = 0; i < 1000; ++i) {
        t2iPipeline t2iPipe;
        t2iPipe.load_model(models_path, device);
        t2iPipe.generate(prompt);
        printMemoryUsage("After generate image.");
        t2iPipe.release();

        for (int j = 0; j < 6; ++j) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            std::string delay_msg = "Release Text2Image Pipeline after " +  std::to_string(j * 10 + 10) + " seconds";
            printMemoryUsage(delay_msg);
        }
        PrintLoadedDlls();
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
