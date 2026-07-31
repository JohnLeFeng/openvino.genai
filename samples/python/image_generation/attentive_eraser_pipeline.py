#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino
import openvino_genai
import numpy as np
from PIL import Image


def read_image(path: str, target_size: tuple[int, int]) -> tuple[openvino.Tensor, bool]:
    image = Image.open(path).convert("RGB")
    resized = image.size != target_size
    if resized:
        image = image.resize(target_size, Image.Resampling.NEAREST)
    return openvino.Tensor(np.array(image)[None]), resized


def read_mask(path: str, target_size: tuple[int, int]) -> tuple[openvino.Tensor, bool]:
    mask = Image.open(path)
    mask = mask.convert("L" if len(mask.getbands()) == 1 else "RGB")
    resized = mask.size != target_size
    if resized:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    mask_data = np.array(mask)
    if mask_data.ndim == 2:
        mask_data = mask_data[..., None]
    return openvino.Tensor(mask_data[None]), resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", metavar="MODEL_DIR")
    parser.add_argument("image", metavar="IMAGE")
    parser.add_argument("mask_image", metavar="MASK_IMAGE")
    parser.add_argument("device", metavar="DEVICE", nargs="?", default="CPU")
    parser.add_argument("seed", metavar="SEED", nargs="?", type=int, default=123)
    args = parser.parse_args()

    pipeline = openvino_genai.InpaintingPipeline(
        args.model_dir,
        args.device,
        inpainting_mode=openvino_genai.InpaintingMode.ATTENTIVE_ERASER,
    )

    attentive_eraser = openvino_genai.AttentiveEraserConfig()
    attentive_eraser.rm_guidance_scale = 9.0
    attentive_eraser.ss_steps = 9

    config = pipeline.get_generation_config()
    target_size = (config.width, config.height)
    image, image_resized = read_image(args.image, target_size)
    mask, mask_resized = read_mask(args.mask_image, target_size)
    if image_resized or mask_resized:
        print(f"Resizing image and mask to {config.width}x{config.height}")

    config.strength = 0.8
    config.num_inference_steps = 50
    config.rng_seed = args.seed
    config.attentive_eraser = attentive_eraser
    pipeline.set_generation_config(config)

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    generated_image = pipeline.generate(
        "",
        image,
        mask,
        callback=callback,
    )
    Image.fromarray(generated_image.data[0]).save("object_removed_image.bmp")


if __name__ == "__main__":
    main()