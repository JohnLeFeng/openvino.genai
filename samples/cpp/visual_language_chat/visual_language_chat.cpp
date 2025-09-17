// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>
#include <windows.h>
#include <codecvt>
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword.c_str() << std::flush);
}

std::string wstringToString(const std::wstring& wstr, UINT codePage = CP_UTF8) {
    int bufferSize = WideCharToMultiByte(codePage, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (bufferSize == 0) {
        return ""; // Handle error
    }
    std::string str(bufferSize - 1, '\0'); // -1 to exclude null terminator from size
    WideCharToMultiByte(codePage, 0, wstr.c_str(), -1, &str[0], bufferSize, nullptr, nullptr);
    return str;
}

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 5) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE> [<Cache type: 0: Model Cache (default), 1: Blob, 2: Weightless Blob>]");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = argv[3];
    std::string model_dir = argv[1];
    int cache_type = (argc == 5) ? std::stoi(argv[4]) : 0;

    ov::AnyMap npu_properties;
    if (device == "NPU") {
        switch (cache_type) {
            case 0:
                std::cout << "[INFO] Use model cache" << std::endl;
                npu_properties = {
                    {"NPU_BYPASS_UMD_CACHING", "NO"},
                    {"NPUW_DEVICES", "NPU"},
                    {"MAX_PROMPT_LEN" , 4096},
                    {"MIN_RESPONSE_LEN", 256},
                    {"NPUW_LLM_PREFILL_CHUNK_SIZE" , 1024},
                    {"NPUW_CACHE_DIR", "npuw-cache"},
                    {"NPUW_PARALLEL_COMPILE", true},
                    {"GENERATE_HINT", "BEST_PERF"}
                };
                break;
            case 1:
                if (std::filesystem::exists(model_dir + "\\openvino_language_model_general.blob")) {
                    std::cout << "[INFO] Use blob" << std::endl;
                    npu_properties = {
                        {"BLOB_PATH",  model_dir + "\\openvino_language_model_general.blob"},
                        {"NPU_BYPASS_UMD_CACHING", "NO"},
                        {"NPUW_DEVICES", "NPU"},
                        {"MAX_PROMPT_LEN" , 4096},
                        {"MIN_RESPONSE_LEN", 256},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE" , 1024},
                        {"NPUW_PARALLEL_COMPILE", true},
                        {"GENERATE_HINT", "BEST_PERF"}
                    };
                }
                else {
                    std::cout << "[INFO] No blob found. Exporting blob." << std::endl;
                    npu_properties = {
                        {"EXPORT_BLOB", "YES"},
                        {"BLOB_PATH", model_dir + "\\openvino_language_model_general.blob"},
                        {"CACHE_MODE", "OPTIMIZE_SPEED"},
                        {"NPU_BYPASS_UMD_CACHING", "NO"},
                        {"NPUW_DEVICES", "NPU"},
                        {"MAX_PROMPT_LEN" , 4096},
                        {"MIN_RESPONSE_LEN", 256},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE" , 1024},
                        {"NPUW_PARALLEL_COMPILE", true},
                        {"GENERATE_HINT", "BEST_PERF"}
                    };
                }
                break;
            case 2:
                if (std::filesystem::exists(model_dir + "\\openvino_language_model_weightless.blob")) {
                    std::cout << "[INFO] Use weightless blob" << std::endl;
                    npu_properties = {
                        {"BLOB_PATH",  model_dir + "\\openvino_language_model_weightless.blob"},
                        {"WEIGHTS_PATH", model_dir + "\\openvino_language_model.bin"},
                        {"NPU_BYPASS_UMD_CACHING", "NO"},
                        {"NPUW_DEVICES", "NPU"},
                        {"MAX_PROMPT_LEN" , 4096},
                        {"MIN_RESPONSE_LEN", 256},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE" , 1024},
                        {"NPUW_PARALLEL_COMPILE", true},
                        {"GENERATE_HINT", "BEST_PERF"}
                    };
                }
                else {
                    std::cout << "[INFO] No blob found. Exporting weightless blob" << std::endl;
                    npu_properties = {
                        {"EXPORT_BLOB", "YES"},
                        {"BLOB_PATH", model_dir + "\\openvino_language_model_weightless.blob"},
                        {"NPU_BYPASS_UMD_CACHING", "NO"},
                        {"NPUW_DEVICES", "NPU"},
                        {"MAX_PROMPT_LEN" , 4096},
                        {"MIN_RESPONSE_LEN", 256},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE" , 1024},
                        {"NPUW_PARALLEL_COMPILE", true},
                        {"GENERATE_HINT", "BEST_PERF"}
                    };
                }
                break;
            default:
                throw std::runtime_error("Wrong cache type");
        }
    }

    ov::AnyMap gpu_properties = {
        {"CACHE_DIR", "gpu-cache"},

    };
    ov::AnyMap device_properties = {
        {"NPU", npu_properties},
        {"GPU", gpu_properties}
    };
    ov::AnyMap properties = {
        {"DEVICE_PROPERTIES", device_properties},
    };

    auto start_t = std::chrono::high_resolution_clock::now();
    ov::genai::VLMPipeline pipe(model_dir, device, properties);
    auto end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Load model time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() * 0.001 * 0.001 << " s" << std::endl;

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 256;

    std::string en_prompt = "What is unusual on this picture?";
    std::wstring cn_prompt = L"这张图片上有什么不寻常的地方?";
    std::wstring zh_prompt = L"這張圖片上有甚麼不尋常的地方?";

    std::cout << "[INFO] EN prompt: " << en_prompt << std::endl;
    start_t = std::chrono::high_resolution_clock::now();
    pipe.generate(en_prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword)
                );
    std::cout << std::endl;
    end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Process EN took: "  << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() * 0.001 * 0.001 << " s" << std::endl;

    std::cout << "[INFO] CN prompt: " << wstringToString(cn_prompt) << std::endl;
    start_t = std::chrono::high_resolution_clock::now();
    pipe.generate(wstringToString(cn_prompt),
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword)
                  );
    std::cout << std::endl;
    end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Process CN took: "  << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() * 0.001 * 0.001 << " s" << std::endl;

    std::cout << "[INFO] ZH prompt: " << wstringToString(zh_prompt) << std::endl;
    start_t = std::chrono::high_resolution_clock::now();
    pipe.generate(wstringToString(zh_prompt),
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword)
                 );
    std::cout << std::endl;
    end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Process ZH took: "  << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() * 0.001 * 0.001 << " s" << std::endl;


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
