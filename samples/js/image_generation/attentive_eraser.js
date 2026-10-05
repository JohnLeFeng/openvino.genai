// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from "node:path";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { InpaintingMode, InpaintingPipeline } from "openvino-genai-node";
import { readImage, saveAsBMP } from "../image_utils.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      "$0 <model_dir> <image> <mask> [device]",
      "Remove a masked object with Attentive Eraser and save the result as a BMP file",
      (yargsBuilder) =>
        yargsBuilder
          .positional("model_dir", {
            type: "string",
            describe: "Path to the exported Stable Diffusion 1.5, 2, or SDXL Base model",
            demandOption: true,
          })
          .positional("image", {
            type: "string",
            describe: "Path to the input image (JPEG, PNG or BMP)",
            demandOption: true,
          })
          .positional("mask", {
            type: "string",
            describe: "Path to the mask image marking the object to remove",
            demandOption: true,
          })
          .positional("device", {
            type: "string",
            describe: "Inference device (CPU or GPU)",
            default: "CPU",
          }),
    )
    .strict()
    .help()
    .parse();

  const pipeline = await InpaintingPipeline(argv.model_dir, argv.device, {
    inpainting_mode: InpaintingMode.ATTENTIVE_ERASER,
  });
  const config = pipeline.getGenerationConfig();
  config.guidance_scale = 1.0;
  config.attentive_eraser = {
    rm_guidance_scale: 9.0,
  };
  pipeline.setGenerationConfig(config);

  const imageTensor = await readImage(argv.image, { batched: true });
  const maskTensor = await readImage(argv.mask, { batched: true });
  const resultTensor = await pipeline.generate("", imageTensor, maskTensor);
  await saveAsBMP("object_removed_image.bmp", resultTensor);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
