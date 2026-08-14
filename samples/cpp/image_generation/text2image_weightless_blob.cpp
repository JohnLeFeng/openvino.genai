// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/core.hpp>
#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"


void blob_export(const std::filesystem::path& root_dir,
                 const std::string blob_folder_name,
                 const std::string& ov_cache_dir,
                 const std::string text_encoder_device,
                 const std::string unet_device,
                 const std::string vae_decoder_device,
                 const int width,
                 const int height
) {
    const auto blob_path = root_dir / blob_folder_name;

    ov::AnyMap npu_props{
        {"PERFORMANCE_HINT", "LATENCY"},
        {"ENABLE_WEIGHTLESS", "YES"},
        {"CACHE_MODE", "OPTIMIZE_SIZE"},
        {"NPU_DEFER_WEIGHTS_LOAD", "YES"},
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_SEPARATE_WEIGHTS_VERSION", "ONE_SHOT"},
    };

    ov::AnyMap properties{
        {"DEVICE_PROPERTIES", ov::AnyMap{{"NPU", npu_props}}},
        {"CACHE_DIR", ov_cache_dir}
    };

    ov::genai::Text2ImagePipeline pipe(root_dir);
    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale);
    pipe.compile(text_encoder_device, unet_device, vae_decoder_device, properties);
    
    pipe.export_model(blob_path);
};

ov::genai::Text2ImagePipeline blob_import(const std::filesystem::path& root_dir,
                 const std::string blob_folder_name,
                 const std::string& ov_cache_dir,
                 const std::string text_encoder_device,
                 const std::string unet_device,
                 const std::string vae_decoder_device,
                 const int width,
                 const int height
) {
    const auto blob_path = root_dir / blob_folder_name;
    const auto unet_blob_path = root_dir / blob_folder_name / "unet";
    
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(root_dir / "unet" / "openvino_model.xml");
    
    ov::AnyMap npu_props{
        {"PERFORMANCE_HINT", "LATENCY"},
        // {"WEIGHTS_PATH", (root_dir / "unet" / "openvino_model.bin").string()}
    };
    npu_props.insert(ov::hint::model(model));

    ov::AnyMap properties{
        {"DEVICE_PROPERTIES", ov::AnyMap{{"NPU", npu_props}}},
        {"blob_path", unet_blob_path.string()},
    };

    return ov::genai::Text2ImagePipeline::stable_diffusion_xl(
        ov::genai::Scheduler::from_config(root_dir / "scheduler" / "scheduler_config.json"),
        ov::genai::CLIPTextModel(root_dir / "text_encoder", text_encoder_device),
        ov::genai::CLIPTextModelWithProjection(root_dir / "text_encoder_2", text_encoder_device),
        ov::genai::UNet2DConditionModel(root_dir / "unet", unet_device, properties),
        ov::genai::AutoencoderKL(root_dir / "vae_decoder", vae_decoder_device)
    );
};

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [ <TXT_ENCODE_DEVICE> <UNET_DEVICE> <VAE_DEVICE> ]");

    const std::string models_path = argv[1], prompt = argv[2];

    std::filesystem::path root_dir = models_path;

    const int width = 960;
    const int height = 512;
    const int number_of_images_to_generate = 1;
    const int number_of_inference_steps_per_image = 20;

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

    blob_export(
        root_dir,
        "blobs",
        ov_cache_dir,
        text_encoder_device,
        unet_device,
        vae_decoder_device,
        width,
        height
    );
    std::cout << "Blob export complete" << std::endl;

    auto pipe = blob_import(
        root_dir,
        "blobs",
        ov_cache_dir,
        text_encoder_device,
        unet_device,
        vae_decoder_device,
        width,
        height
    );

    std::cout << "Blob import complete" << std::endl;

    for (int imagei = 0; imagei < number_of_images_to_generate; imagei++) {
        std::cout << "Generating image " << imagei << std::endl;

        ov::Tensor image = pipe.generate(prompt,
                                         ov::genai::width(width),
                                         ov::genai::height(height),
                                         ov::genai::num_inference_steps(number_of_inference_steps_per_image),
                                         ov::genai::callback(progress_bar));

        imwrite("image_" + std::to_string(imagei) + ".bmp", image, true);
    }
    std::cout << "Image generation complete" << std::endl;

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
