// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <queue>
#include <chrono>
#include <thread>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"


void decode_latent(std::queue<ov::Tensor>* intermediate_latent, std::function<ov::Tensor(const ov::Tensor)>& decoder, int32_t num_steps) {
    int i = 0;
    while (i < (num_steps - 1)) {
        std::cout << "intermediate_latent size: " << intermediate_latent->size();
        if (!intermediate_latent->empty()) {
            std::cout << "Do decoder for latent " << i + 1 << " ..." << std::endl;
            ov::Tensor intermediate_image = decoder(intermediate_latent->front());
            imwrite("intermediate_image_" + std::to_string(i+1) + ".bmp", intermediate_image, true);
            std::cout << "Saved intermediate result " << i + 1 << std::endl;
            intermediate_latent->pop();
            i++;
        }
    }
}

int32_t main(int32_t argc, char* argv[]) try {

    // std::thread my_thread(greet, user_name, 10);

    OPENVINO_ASSERT(argc >= 3 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [ <TXT_ENCODE_DEVICE> <UNET_DEVICE> <VAE_DEVICE> ]");

    const std::string models_path = argv[1], prompt = argv[2];

    std::filesystem::path root_dir = models_path;

    const int width = 960;
    const int height = 512;
    const int number_of_images_to_generate = 1;
    const int number_of_inference_steps_per_image = 12;

    // Set devices to command-line args if specified, otherwise default to CPU.
    // Note that these can be set to CPU, GPU, or NPU.
    const std::string text_encoder_device = (argc > 3) ? argv[3] : "CPU";
    const std::string unet_device = (argc > 4) ? argv[4] : "NPU";
    const std::string vae_decoder_device = (argc > 5) ? argv[5] : "GPU";

    std::cout << "text_encoder_device: " << text_encoder_device << std::endl;
    std::cout << "unet_device: " << unet_device << std::endl;
    std::cout << "vae_decoder_device: " << vae_decoder_device << std::endl;

    // this is the path to where compiled models will get cached
    // (so that the 'compile' method run much faster 2nd+ time)
    std::string ov_cache_dir = "./cache";

    //
    // Step 1: Create the initial Text2ImagePipeline, given the model path
    //
    std::unique_ptr<ov::genai::Text2ImagePipeline> t2i_pipeline;
    t2i_pipeline = std::make_unique<ov::genai::Text2ImagePipeline>(models_path);

    //
    // Step 2: Reshape the pipeline given number of images, height, width and guidance scale.
    //
    t2i_pipeline->reshape(1, height, width, t2i_pipeline->get_generation_config().guidance_scale);

    //
    // Step 3: Compile the pipeline with the specified devices, and properties (like cache dir)
    //
    ov::AnyMap properties = {ov::cache_dir(ov_cache_dir)};

    // Note that if there are device-specific properties that are needed, they can
    // be added using ov::device::properties groups, like this:
    //ov::AnyMap properties = {ov::device::properties("CPU", ov::cache_dir("cpu_cache")),
    //                         ov::device::properties("GPU", ov::cache_dir("gpu_cache")),
    //                         ov::device::properties("NPU", ov::cache_dir("npu_cache"))};

    t2i_pipeline->compile(text_encoder_device, unet_device, vae_decoder_device, properties);

    std::queue<ov::Tensor>* intermediate_latent = nullptr;
    intermediate_latent = new std::queue<ov::Tensor>();

    std::function<ov::Tensor(const ov::Tensor)> decoder_func = 
        [&](const ov::Tensor& latent) {
            return t2i_pipeline->decode(latent);
        };

    std::thread decode_thread(decode_latent, std::ref(intermediate_latent), decoder_func, number_of_inference_steps_per_image);

    auto callback = [&](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
        std::cout << "Image generation step: " << step + 1 << " / " << num_steps << std::endl;
        if (step < num_steps - 1) {
            std::cout << "Add latent into intermediate latent buffer." << std::endl;
            intermediate_latent->push(latent);
            std::cout << "size: " << intermediate_latent->size() << std::endl;
        } else if (step == num_steps - 1) {
            decode_thread.join();
        }
        return false;
    };
    
    // Step 4: Use the Text2ImagePipeline to generate 'number_of_images_to_generate' images.
    //
    std::cout << "Generating image ..." << std::endl;

    auto start_t = std::chrono::high_resolution_clock::now();
    ov::Tensor image = t2i_pipeline->generate(prompt,
                                        ov::genai::num_inference_steps(number_of_inference_steps_per_image),
                                        ov::genai::callback(callback));
    auto end_t = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Inference time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() * 0.001 * 0.001 << " s" << std::endl;
    imwrite("image.bmp", image, true);


    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
