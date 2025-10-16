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
import queue
import logging
import argparse
import threading


import openvino_genai
from PIL import Image


from memory_monitor import MemMonitorWrapper, MemoryUnit


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('text_encoder_device', nargs='?', default='CPU')
    parser.add_argument('unet_device', nargs='?', default='NPU')
    parser.add_argument('vae_decoder_device', nargs='?', default='GPU')
    args = parser.parse_args()

    width = 960
    height = 512

    num_inference_steps = 12

    # use_callback = True
    use_callback = False


    log.info(f"text encoder is running on {args.text_encoder_device}.")
    log.info(f"unet is running on {args.unet_device}.")
    log.info(f"vae decoder is running on {args.vae_decoder_device}.")

    ov_cache_dir = "./cache"
    memory_monitor = MemMonitorWrapper()
    memory_monitor.memory_unit = MemoryUnit.GiB
    memory_monitor.create_monitors()
    memory_monitor.start()

    memory_unit = memory_monitor.memory_unit.value

    s_time = time.time()

    pipe = openvino_genai.Text2ImagePipeline(args.model_dir)

    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)

    properties = {"CACHE_DIR": ov_cache_dir}

    pipe.compile(args.text_encoder_device, args.unet_device, args.vae_decoder_device, config=properties)

    e_time = time.time()
    log.info(f"Load model time: {e_time - s_time}.")

    memory_monitor.stop_and_collect_data("")
    max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = memory_monitor.get_data()

    log.info ("Max RSS Memory consumption: %.2f %s, Max RSS Memory increase: %.2f %s",
        max_rss_mem_consumption, memory_unit, max_rss_mem_increase, memory_unit
    )
    log.info ("Max System Memory consumption: %.2f %s, Max System Memory increase: %.2f %s",
        max_sys_mem_consumption, memory_unit, max_sys_mem_increase, memory_unit
    )

    memory_monitor.start()
    if use_callback:
        intermediate_latent = queue.Queue()

        def decode_latent(intermediate_latent, decoder, num_steps):
            i = 0
            while i < num_steps:
                try:
                    latent = intermediate_latent.get_nowait()
                    log.info("Do decoder for latent %s ...", i + 1)
                    image_tensor = decoder(latent)
                    image = Image.fromarray(image_tensor.data[0])
                    image.save("image_{}.bmp".format(i + 1))
                    log.info ("Saved intermediate result %s.", i + 1)
                    i = i + 1
                except queue.Empty:
                    continue

        thread = threading.Thread(
            target=decode_latent,
            args=(intermediate_latent, pipe.decode, num_inference_steps - 1)  # Pass "Task1" as name and 3 as delay
        )

        def callback(step, num_steps, latent):
            log.info(f"Image generation step: {step + 1} / {num_steps}")
            if step < num_steps - 1:
                log.info("Add latent into intermediate latent buffer.")
                intermediate_latent.put(latent)
            else:
                log.info ("Skip, let pipeline do decode.")
            return False
    
        s_time = time.time()

        thread.start()
        
        image_tensor = pipe.generate(
            args.prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            callback = callback,
            guidance_scale=8.0
        )
        
        e_time = time.time()
        log.info(f"Inference time: {e_time - s_time}.")

        image = Image.fromarray(image_tensor.data[0])
        image.save("image.bmp")

        thread.join()
    else:
        s_time = time.time()

        image_tensor = pipe.generate(
            args.prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=8.0
        )
        
        e_time = time.time()
        log.info(f"Inference time: {e_time - s_time}.")

        image = Image.fromarray(image_tensor.data[0])
        image.save("image.bmp")
    
    memory_monitor.stop_and_collect_data("")
    max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = memory_monitor.get_data()

    log.info ("Max RSS Memory consumption: %.2f %s, Max RSS Memory increase: %.2f %s",
        max_rss_mem_consumption, memory_unit, max_rss_mem_increase, memory_unit
    )
    log.info ("Max System Memory consumption: %.2f %s, Max System Memory increase: %.2f %s",
        max_sys_mem_consumption, memory_unit, max_sys_mem_increase, memory_unit
    )

if '__main__' == __name__:
    sys.exit(main())