#
# This script demonstrates how to use the attentive eraser pipeline for object removal in images.
# Sample code are from https://github.com/Alibaba-VELLDEPTH/AttentiveEraser/tree/master#usage-example-in--diffusers
#

import sys
import torch
from pathlib import Path
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import argparse
from types import MethodType


dtype = torch.float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


MODEL_CONFIG = {
    "SD2": {
        # "model_name": "stabilityai/stable-diffusion-2-1-base",
        # "model_name": "Manojb/stable-diffusion-2-1-base",
        "model_name": "C:\\Users\\John\\Documents\\ov_genai_fork3\\stable-diffusion-2-1-base",
        "pipeline": StableDiffusionInpaintPipeline,
        "custom_pipeline": "./atten_eraser_pipeline/pipeline_inp.py",
        "height": 512,
        "width": 512,
        "mask_blur_kernel": 7,
        "source_image_path": "https://raw.githubusercontent.com/Alibaba-VELLDEPTH/AttentiveEraser/refs/heads/master/examples/img/an.png",
        "mask_path": "https://raw.githubusercontent.com/Alibaba-VELLDEPTH/AttentiveEraser/refs/heads/master/examples/mask/an_mask.png",
        "AAS_start_step": 0,  # AAS start step
        #
        # 0~5down,6mid,7~15up /layer that starting AAS
        # From: https://github.com/Alibaba-VELLDEPTH/AttentiveEraser/blob/master/notebook/Attentive_Eraser_SIP.ipynb
        #
        "AAS_start_layer": 7,  # AAS start layer,
        "AAS_end_layer": 16,  # AAS end layer,
        "ss_steps": 9,  # similarity suppression steps
        "ss_scale": 0.3,  # similarity suppression scale
        "rm_guidance_scale": 9.0,  # removal guidance scale
        "export_dir": "./sd2_atten_eraser_ov/unet",  # Output directory for OpenVINO IR
        "output_dir": "./sd2_atten_eraser_results/torch",  # Output directory for generated images
    },
    "SDXL": {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": DiffusionPipeline,
        "custom_pipeline": "./atten_eraser_pipeline/pipeline_stable_diffusion_xl_attentive_eraser.py",
        "height": 1024,
        "width": 1024,
        "mask_blur_kernel": 77,
        "source_image_path": "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024.png",
        "mask_path": "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024_mask.png",
        "AAS_start_step": 0,  # AAS start step
        "AAS_start_layer": 34,  # AAS start layer
        "AAS_end_layer": 70,  # AAS end layer
        "ss_steps": 9,  # similarity suppression steps
        "ss_scale": 0.3,  # similarity suppression scale
        "rm_guidance_scale": 9.0,  # removal guidance scale
        "export_dir": "./sdxl_atten_eraser_ov/unet",  # Output directory for OpenVINO IR
        "output_dir": "./sdxl_atten_eraser_results/torch",  # Output directory for generated images
    },
}

MODEL_CONFIG["SD15"] = dict(MODEL_CONFIG["SD2"])  # SD1.5 shares SD2's config (independent copy)
MODEL_CONFIG["SD15"]["model_name"] = "runwayml/stable-diffusion-v1-5"
MODEL_CONFIG["SD15"]["export_dir"] = "./sd15_atten_eraser_ov/unet"  # Output directory for OpenVINO IR
MODEL_CONFIG["SD15"]["output_dir"] = "./sd15_atten_eraser_results/torch"  # Output directory for generated images

def parse_args(argv=None) -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Attentive Eraser with optional OpenVINO conversion")
    parser.add_argument(
        "--model_type", type=str, choices=["SD15", "SD2", "SDXL"], default="SDXL", help="Model to use (default: SDXL)"
    )
    parser.add_argument("--convert-unet", action="store_true", help="Convert AAS-modified UNet to OpenVINO IR")
    parser.add_argument(
        "--export-model-only",
        action="store_true",
        help="Run one pipeline iteration to initialize AAS, then export only the OpenVINO UNet",
    )
    parser.add_argument("--save-image", action="store_true", help="Save the generated inpainted image as PNG")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate denoising steps")
    parser.add_argument(
        "--intermediate-steps", type=int, default=1, help="Save intermediate result every N steps (default: 1)"
    )
    parser.add_argument(
        "--use-single-softmax-output-gating",
        action="store_true",
        help="Use experimental single-softmax output gating (default: False, use original dual-softmax AAS)",
    )
    return parser.parse_args(argv)


def create_latents_callback(save_dir, step_interval):
    """Create a callback function to save intermediate denoising steps.

    Args:
        save_dir: Directory to save intermediate images
        step_interval: Save every N steps

    Returns:
        Callback function for use with diffusers pipeline
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    def latents_callback(pipe, step, timestep, callback_kwargs):
        """Save intermediate latents as decoded images."""
        if step % step_interval == 0:
            # Extract the latents at this specific step
            latents = callback_kwargs["latents"]

            # SDXL's VAE overflows to NaN (black image) when decoding in fp16.
            # The normal pipeline upcasts the VAE to fp32 before the final decode;
            # do the same here for the intermediate decode, then restore.
            vae = pipe.vae
            orig_vae_dtype = vae.dtype
            needs_upcast = orig_vae_dtype == torch.float16
            if needs_upcast:
                vae = vae.to(dtype=torch.float32)

            with torch.no_grad():
                # Scale and decode the latents into an image
                scaled = latents.to(vae.dtype) / vae.config.scaling_factor
                image = vae.decode(scaled).sample

            if needs_upcast:
                vae.to(dtype=orig_vae_dtype)

            # Process and save the image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            pil_image = pipe.numpy_to_pil(image)[0]

            # Save the intermediate result with step index in filename
            img_path = Path(save_dir) / f"step_{step:02d}.png"
            pil_image.save(str(img_path))

        return callback_kwargs

    return latents_callback


def _find_aas_editor(unet):
    """Recover the AAS editor that the pipeline monkey-patched onto the UNet.

    ``regiter_attention_editor_diffusers`` replaces each ``Attention.forward``
    with a closure that references the stateful ``editor`` object. We pull that
    object back out of the closure so we can put it into an AAS-active step
    before tracing.
    """
    for module in unet.modules():
        fwd = module.__dict__.get("forward", None)
        code = getattr(fwd, "__code__", None)
        closure = getattr(fwd, "__closure__", None)
        if code is not None and closure is not None and "editor" in code.co_freevars:
            return closure[code.co_freevars.index("editor")].cell_contents
    return None


def _runtime_aas_masked_attention(
    self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, is_mask_attn, mask, **kwargs
):
    batch = q.shape[0] // num_heads

    def attention_output(attention_scores):
        output = torch.einsum("h i j, h j d -> h i d", attention_scores, v)
        return (
            output.reshape(1, num_heads, batch * q.shape[1], v.shape[2])
            .permute(0, 2, 1, 3)
            .reshape(1, batch * q.shape[1], num_heads * v.shape[2])
        )

    if not is_mask_attn:
        return attention_output(attn)

    mask_flatten = mask.reshape(-1)
    key_mask = mask.reshape(1, 1, -1)
    mask_penalty = key_mask.masked_fill(key_mask == 1, torch.finfo(sim.dtype).min)
    
    # Use single-softmax output gating or original dual-softmax AAS.
    if hasattr(self, "use_single_softmax_output_gating") and self.use_single_softmax_output_gating:
        # ========== EXPERIMENTAL: Single-softmax output gating ==========
        sim_masked = sim + mask_penalty
        attn_weights = sim_masked.softmax(dim=2)
        out_base = attention_output(attn_weights)
        out_fg = out_base
        out_bg = out_base
        
        # Apply mask_suppression only during ss_steps
        if self.runtime_cur_step <= self.runtime_ss_steps:
            mask_suppression = 1.0 - mask_flatten * (1.0 - self.ss_scale)
            out_fg = out_base * mask_suppression.unsqueeze(-1).unsqueeze(0)
    else:
        # ========== ORIGINAL APPROACH: Dual-softmax with ss_scale*sim ==========
        sim_bg = sim + mask_penalty
        sim_fg = self.ss_scale * sim + mask_penalty
        out_bg = attention_output(sim_bg.softmax(dim=2))
        out_fg = attention_output(sim_fg.softmax(dim=2))
        out_fg = torch.where(self.runtime_cur_step <= self.runtime_ss_steps, out_fg, out_bg)
    
    return torch.cat([out_fg, out_bg], dim=0)


def convert_unet_to_openvino(
    unet, export_dir="./sdxl_atten_eraser_ov/unet", model_type="SDXL", height=1024, width=1024
):
    """Convert the AAS-modified UNet to OpenVINO IR (openvino_model.xml/.bin).

    Supports both SDXL and SD1.5/SD2. The two families differ in two ways that
    matter for tracing:

    * SDXL's UNet expects ``added_cond_kwargs`` (a dict with ``text_embeds`` and
      ``time_ids``); SD1.5/SD2 do not. For SDXL we flatten that dict into
      explicit tensors so they become real graph inputs.
    * The per-resolution AAS masks differ: SDXL (1024px) uses 16/32/64/128 while
      SD1.5/SD2 (512px) uses 8/16/32/64. Both are recomputed from the ``mask``
      input inside ``forward`` so the exported IR works for ANY mask.

    NOTE: AAS does not add weights - it monkey-patches each ``Attention.forward``
    with a stateful editor whose behavior depends on Python-level step/layer
    counters and data-dependent branches. Tracing FREEZES that logic: the IR
    reflects only the code path taken at trace time. To capture the AAS path we
    reset the editor to its first AAS-active step before converting.

    The AAS ``mask``, ``cur_step``, and ``ss_steps`` values are exposed as real
    model inputs. The AAS layer range remains fixed at conversion time.
    """
    import openvino as ov

    # The AAS editors look up per-resolution masks via a single dimension
    # ``int(np.sqrt(q.shape[1]))``, which is only valid for square latents. The
    # mask derivation below keys attributes off ``height`` alone for the same
    # reason. Fail loudly on non-square requests instead of emitting a subtly
    # wrong IR. (Full non-square support would require reworking the pipelines.)
    if height != width:
        raise ValueError(
            f"convert_unet_to_openvino currently supports only square latents "
            f"(height == width); got height={height}, width={width}."
        )

    is_sdxl = model_type == "SDXL"
    unet = unet.eval()
    p_dtype = next(unet.parameters()).dtype
    p_device = next(unet.parameters()).device

    # Determine the added-conditioning requirement from the model itself rather
    # than trusting the ``model_type`` string: SDXL's UNet has
    # ``addition_embed_type == "text_time"`` and therefore needs the extra
    # ``text_embeds``/``time_ids`` inputs; SD1.5/SD2 do not. Detecting it here
    # means a wrong/omitted ``model_type`` can't silently produce a broken IR.
    needs_added_cond = getattr(unet.config, "addition_embed_type", None) == "text_time"
    if needs_added_cond != is_sdxl:
        print(
            f"Warning: model_type={model_type!r} but UNet addition_embed_type="
            f"{getattr(unet.config, 'addition_embed_type', None)!r}; following the model."
        )
    is_sdxl = needs_added_cond

    # Put the AAS editor into an active step so the trace exercises the AAS path
    # (after generation it sits past its last step and would trace plain attention).
    editor = _find_aas_editor(unet)
    if editor is not None:
        editor.reset()
        editor.cur_step = editor.start_step
        editor.cur_att_layer = 0
        editor.attn_batch = MethodType(_runtime_aas_masked_attention, editor)
        print(f"AAS editor found; tracing at cur_step={editor.cur_step} (AAS active).")
    else:
        print("Warning: no AAS editor found on UNet; exporting the plain attention path.")

    def _set_editor_masks(editor, mask):
        # Feed the AAS mask as a real graph INPUT instead of a baked constant.
        # Recomputing the per-resolution masks from `mask` here keeps them tied
        # to the input node, so the exported IR works for ANY mask, not just the
        # one used during generation. (kernels mirror AAS_XL/AAS_Base.__init__)
        #
        # Kernel sizes use the constant height/width (not mask.shape) so that
        # max_pool2d's kernel stays a compile-time constant for OpenVINO.
        editor.mask = mask
        # The AAS editor attends at four spatial resolutions equal to the latent
        # size and its halvings: latent, latent/2, latent/4, latent/8. That is
        # 8/16/32/64 for SD1.5/SD2 (512px) and 16/32/64/128 for SDXL (1024px).
        # Deriving them from height/width makes this correct for either family
        # without a model-type branch. Kernel sizes stay compile-time constant
        # (from height/width, not mask.shape) so max_pool2d traces cleanly.
        for res in (height // 8, height // 16, height // 32, height // 64):
            setattr(
                editor,
                f"mask_{res}",
                F.max_pool2d(mask, (height // res, width // res)).round().squeeze(0).squeeze(0),
            )

    class UNetWrapperSDXL(torch.nn.Module):
        def __init__(self, unet, editor):
            super().__init__()
            self.unet = unet
            self.editor = editor

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, mask, cur_step, ss_steps):
            if self.editor is not None:
                _set_editor_masks(self.editor, mask)
                self.editor.runtime_cur_step = cur_step
                self.editor.runtime_ss_steps = ss_steps
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    class UNetWrapperSD(torch.nn.Module):
        def __init__(self, unet, editor):
            super().__init__()
            self.unet = unet
            self.editor = editor

        def forward(self, sample, timestep, encoder_hidden_states, mask, cur_step, ss_steps):
            if self.editor is not None:
                _set_editor_masks(self.editor, mask)
                self.editor.runtime_cur_step = cur_step
                self.editor.runtime_ss_steps = ss_steps
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

    latent_h, latent_w = height // 8, width // 8
    batch = 2  # classifier-free guidance doubles the batch
    cross_attention_dim = unet.config.cross_attention_dim  # 2048 SDXL, 1024 SD2, 768 SD1.5

    example_input = {
        "sample": torch.randn(batch, unet.config.in_channels, latent_h, latent_w, dtype=p_dtype, device=p_device),
        "timestep": torch.tensor(1, dtype=torch.int64, device=p_device),
        "encoder_hidden_states": torch.randn(batch, 77, cross_attention_dim, dtype=p_dtype, device=p_device),
    }
    if is_sdxl:
        example_input["text_embeds"] = torch.randn(batch, 1280, dtype=p_dtype, device=p_device)
        example_input["time_ids"] = torch.randn(batch, 6, dtype=p_dtype, device=p_device)
    # AAS mask input (full resolution, values in {0, 1}); downsampled inside forward.
    example_input["mask"] = torch.randint(0, 2, (1, 1, height, width)).to(dtype=p_dtype, device=p_device)
    example_input["cur_step"] = torch.tensor(0, dtype=torch.int64, device=p_device)
    example_input["ss_steps"] = torch.tensor(9, dtype=torch.int64, device=p_device)

    wrapper = (UNetWrapperSDXL if is_sdxl else UNetWrapperSD)(unet, editor)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)

    export_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, export_dir / "openvino_model.xml")
    print(f"UNet converted to OpenVINO IR at: {export_dir.resolve()}")
    return ov_model


def preprocess_image(image_path, device, height=1024, width=1024):
    image = to_tensor((load_image(image_path)))
    image = image.unsqueeze_(0).float()  # [0,1] range (expected by diffusers)
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (height, width))
    image = image.to(dtype).to(device)
    return image


def preprocess_mask(mask_path, device, height=1024, width=1024, kernel_size=77):
    mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert("L"))))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (height, width))
    mask = gaussian_blur(mask, kernel_size=(kernel_size, kernel_size))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)
    return mask


def save_as_jpg(source_image_path, mask_path, source_out="source_image.png", mask_out="mask.png"):
    """Download the source image and mask (URL or local path) and save them as .png."""
    # Convert to RGB before saving (PNG compatible).
    load_image(source_image_path).convert("RGB").save(source_out, "PNG")
    load_image(mask_path).convert("RGB").save(mask_out, "PNG")
    print(f"Saved source image to {source_out} and mask to {mask_out}")


def main():
    args = parse_args()
    if args.model_type not in MODEL_CONFIG:
        print(f"Error: Invalid model_type '{args.model_type}'. Must be one of {list(MODEL_CONFIG.keys())}.")
        return 1
    model_config = MODEL_CONFIG[args.model_type]
    export_dir = Path(model_config["export_dir"])
    output_dir = Path(model_config["output_dir"])
    intermediate_dir = output_dir / "intermediate_results"
    export_model_only = args.export_model_only
    convert_unet = args.convert_unet or export_model_only
    save_image = args.save_image and not export_model_only
    save_intermediate = args.save_intermediate and not export_model_only
    intermediate_steps = args.intermediate_steps
    num_inference_steps = 1 if export_model_only else 50
    strength = 1.0 if export_model_only else 0.8

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
    )
    pipeline = (
        model_config["pipeline"]
        .from_pretrained(
            model_config["model_name"],
            custom_pipeline=model_config["custom_pipeline"],
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
    )

    prompt = ""  # Set prompt to null
    seed = 123
    generator = torch.Generator(device=device).manual_seed(seed)
    source_image_path = model_config["source_image_path"]
    mask_path = model_config["mask_path"]
    if save_image:
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        source_out = output_dir / "source_image.png"
        mask_out = output_dir / "mask.png"
        save_as_jpg(source_image_path, mask_path, source_out=source_out, mask_out=mask_out)
    source_image = preprocess_image(source_image_path, device, model_config["height"], model_config["width"])
    mask = preprocess_mask(
        mask_path,
        device,
        model_config["height"],
        model_config["width"],
        model_config["mask_blur_kernel"],
    )

    # Create step callback if saving intermediates
    callback_on_step_end = None
    callback_on_step_end_tensor_inputs = None
    if save_intermediate:
        callback_on_step_end = create_latents_callback(intermediate_dir, intermediate_steps)
        callback_on_step_end_tensor_inputs = ["latents"]

    image = pipeline(
        prompt=prompt,
        image=source_image,
        mask_image=mask,
        height=model_config["height"],
        width=model_config["width"],
        AAS=True,  # enable AAS
        strength=strength,  # inpainting strength
        rm_guidance_scale=model_config["rm_guidance_scale"],  # removal guidance scale
        ss_steps=model_config["ss_steps"],  # similarity suppression steps
        ss_scale=model_config["ss_scale"],  # similarity suppression scale
        AAS_start_step=model_config["AAS_start_step"],  # AAS start step
        AAS_start_layer=model_config["AAS_start_layer"],  # AAS start layer
        AAS_end_layer=model_config["AAS_end_layer"],  # AAS end layer
        use_single_softmax_output_gating=args.use_single_softmax_output_gating,
        num_inference_steps=num_inference_steps,  # AAS_end_step = int(strength*num_inference_steps)
        generator=generator,
        guidance_scale=1,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    ).images[0]

    if not export_model_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Add tag to filename indicating which AAS method was used
        aas_tag = "single_softmax_output_gating" if args.use_single_softmax_output_gating else "original_aas"
        output_image_path = output_dir / f"result_{aas_tag}.png"
        image.save(output_image_path)
        print(f"Object removal completed. Image saved to {output_image_path}")

    if save_intermediate:
        print(f"Intermediate denoising steps saved to {intermediate_dir} (every {intermediate_steps} steps)")

    # Convert the AAS-modified UNet to OpenVINO IR after generation (if --convert-unet flag is set).
    if convert_unet:
        convert_unet_to_openvino(
            pipeline.unet,
            export_dir=export_dir,
            height=model_config["height"],
            width=model_config["width"],
            model_type=args.model_type,
        )
    else:
        print("Skipping UNet conversion (use --convert-unet to enable)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
