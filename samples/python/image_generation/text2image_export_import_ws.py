# 
# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import sys
import time
import logging
import argparse
import threading


import openvino_genai
from PIL import Image
from pathlib import Path


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('text_encoder_device', nargs='?', default='CPU')
    parser.add_argument('unet_device', nargs='?', default='NPU')
    parser.add_argument('vae_decoder_device', nargs='?', default='CPU')
    args = parser.parse_args()

    width = 960
    height = 512

    log.info(f"text encoder is running on {args.text_encoder_device}.")
    log.info(f"unet is running on {args.unet_device}.")
    log.info(f"vae decoder is running on {args.vae_decoder_device}.")

    ov_cache_dir = "./cache"

    root_dir = Path(args.model_dir)

    s_time = time.time()

    pipe = openvino_genai.Text2ImagePipeline(root_dir)

    config = pipe.get_generation_config()
    config.guidance_scale = 0.5
    pipe.set_generation_config(config)

    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)

    properties = {
        "DEVICE_PROPERTIES":
        {
            "GPU": {"CACHE_DIR": ov_cache_dir},
            "NPU": {'PERFORMANCE_HINT': 'LATENCY', 'NPU_PLATFORM': 'NPU5000',
                    'NPU_COMPILER_TYPE': 'MLIR', 'NPU_WEIGHTLESS_BLOB': 'YES', 
                    'NPU_SEPARATE_WEIGHTS_VERSION': 'ONE_SHOT','NPU_DEFER_WEIGHTS_LOAD': 'YES'
            }
        }
    }

    pipe.compile(args.text_encoder_device, args.unet_device, args.vae_decoder_device, config=properties )
    # pipe.compile(args.text_encoder_device, args.unet_device, args.vae_decoder_device)

    e_time = time.time()
    log.info(f"Load model time: {e_time - s_time}.")

    blob_path = root_dir / "blobs"

    s_time = time.time()
    
    pipe.export_model(blob_path)

    e_time = time.time()
    log.info(f"Export model time: {e_time - s_time}.")
    del pipe
    
    s_time = time.time()

    properties_load_ws = {
        "DEVICE_PROPERTIES":
        {
            "GPU": {"CACHE_DIR": ov_cache_dir},
            "NPU": {"WEIGHTS_PATH": (root_dir / "unet" / "openvino_model.bin").as_posix()}
        }
    }


    text_encoder = openvino_genai.CLIPTextModel(root_dir / "text_encoder", args.text_encoder_device, blob_path=blob_path / "text_encoder")
    text_encoder_2 = openvino_genai.CLIPTextModelWithProjection(root_dir / "text_encoder_2", args.text_encoder_device, blob_path=blob_path / "text_encoder_2")
    unet = openvino_genai.UNet2DConditionModel(root_dir / "unet", args.unet_device, blob_path=blob_path / "unet", config=properties_load_ws)
    vae = openvino_genai.AutoencoderKL(root_dir / "vae_decoder", args.vae_decoder_device, blob_path=blob_path)

    
    pipe = openvino_genai.Text2ImagePipeline.stable_diffusion_xl(
        scheduler=openvino_genai.Scheduler.from_config(root_dir / "scheduler" / "scheduler_config.json"),
        clip_text_model=text_encoder,
        clip_text_model_with_projection=text_encoder_2,
        unet=unet,
        vae=vae,
    )
    # pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)
    e_time = time.time()
    log.info(f"Load blob time: {e_time - s_time}.")

    # config = pipe.get_generation_config()
    # config.height = height
    # config.width = width
    # config.num_images_per_prompt = 1
    # pipe.set_generation_config(config)

    s_time = time.time()

    image_tensor = pipe.generate(
        args.prompt,
        width=960,
        height=512,
        num_inference_steps=12,
        num_images_per_prompt=1,
        guidance_scale=0.5
    )
    
    e_time = time.time()
    log.info(f"Inference time: {e_time - s_time}.")

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    sys.exit(main())