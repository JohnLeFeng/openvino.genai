#
# This script demonstrates how to use the attentive eraser pipeline for object removal in images.
# Sample code are from https://github.com/Alibaba-VELLDEPTH/AttentiveEraser/tree/master#usage-example-in--diffusers
#

import torch
from pathlib import Path
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur

dtype = torch.float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser",
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)


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


def convert_unet_to_openvino(unet, output_dir="./sdxl_atten_eraser_ov/unet", height=1024, width=1024):
    """Convert the AAS-modified SDXL UNet to OpenVINO IR (openvino_model.xml/.bin).

    NOTE: AAS does not add weights - it monkey-patches each ``Attention.forward``
    with a stateful editor whose behavior depends on Python-level step/layer
    counters and data-dependent branches. Tracing FREEZES that logic: the IR
    reflects only the code path taken at trace time. To capture the AAS path we
    reset the editor to its first AAS-active step before converting.

    The AAS ``mask`` is exposed as a real model INPUT (not baked as a constant),
    so the exported IR works for any mask. Only the step/layer switches (e.g.
    ``ss_steps``) remain frozen at trace time.

    SDXL's UNet also expects ``added_cond_kwargs`` (a dict with ``text_embeds``
    and ``time_ids``); we wrap it to flatten that dict into explicit tensors.
    """
    import openvino as ov

    unet = unet.eval()
    p_dtype = next(unet.parameters()).dtype
    p_device = next(unet.parameters()).device

    # Put the AAS editor into an active step so the trace exercises the AAS path
    # (after generation it sits past its last step and would trace plain attention).
    editor = _find_aas_editor(unet)
    if editor is not None:
        editor.reset()
        editor.cur_step = editor.start_step
        editor.cur_att_layer = 0
        print(f"AAS editor found; tracing at cur_step={editor.cur_step} (AAS active).")
    else:
        print("Warning: no AAS editor found on UNet; exporting the plain attention path.")

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet, editor):
            super().__init__()
            self.unet = unet
            self.editor = editor

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, mask):
            # Feed the AAS mask as a real graph INPUT instead of a baked constant.
            # Recomputing the per-resolution masks from `mask` here keeps them tied
            # to the input node, so the exported IR works for ANY mask, not just the
            # one used during generation. (kernels mirror AAS_XL.__init__)
            #
            # Kernel sizes use the constant height/width (not mask.shape) so that
            # max_pool2d's kernel stays a compile-time constant for OpenVINO.
            if self.editor is not None:
                self.editor.mask = mask
                self.editor.mask_16 = F.max_pool2d(mask, (height // 16, width // 16)).round().squeeze().squeeze()
                self.editor.mask_32 = F.max_pool2d(mask, (height // 32, width // 32)).round().squeeze().squeeze()
                self.editor.mask_64 = F.max_pool2d(mask, (height // 64, width // 64)).round().squeeze().squeeze()
                self.editor.mask_128 = F.max_pool2d(mask, (height // 128, width // 128)).round().squeeze().squeeze()
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    latent_h, latent_w = height // 8, width // 8
    batch = 2  # classifier-free guidance doubles the batch
    cross_attention_dim = unet.config.cross_attention_dim          # 2048 for SDXL base

    example_input = {
        "sample": torch.randn(batch, unet.config.in_channels, latent_h, latent_w, dtype=p_dtype, device=p_device),
        "timestep": torch.tensor(1, dtype=torch.int64, device=p_device),
        "encoder_hidden_states": torch.randn(batch, 77, cross_attention_dim, dtype=p_dtype, device=p_device),
        "text_embeds": torch.randn(batch, 1280, dtype=p_dtype, device=p_device),
        "time_ids": torch.randn(batch, 6, dtype=p_dtype, device=p_device),
        # AAS mask input (full resolution, values in {0, 1}); downsampled inside forward.
        "mask": torch.randint(0, 2, (1, 1, height, width)).to(dtype=p_dtype, device=p_device),
    }

    wrapper = UNetWrapper(unet, editor)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output_dir / "openvino_model.xml")
    print(f"UNet converted to OpenVINO IR at: {output_dir.resolve()}")
    return ov_model


def preprocess_image(image_path, device):
    image = to_tensor((load_image(image_path)))
    image = image.unsqueeze_(0).float() * 2 - 1 # [0,1] --> [-1,1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(dtype).to(device)
    return image

def preprocess_mask(mask_path, device):
    mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)
    return mask

def save_as_jpg(source_image_path, mask_path, source_out="source_image.jpg", mask_out="mask.jpg"):
    """Download the source image and mask (URL or local path) and save them as .jpg."""
    # JPEG has no alpha channel, so convert both to RGB before saving.
    load_image(source_image_path).convert("RGB").save(source_out, "JPEG", quality=95)
    load_image(mask_path).convert("RGB").save(mask_out, "JPEG", quality=95)
    print(f"Saved source image to {source_out} and mask to {mask_out}")

prompt = "" # Set prompt to null
seed=123 
generator = torch.Generator(device=device).manual_seed(seed)
source_image_path = "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024.png"
mask_path = "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024_mask.png"
save_as_jpg(source_image_path, mask_path)
source_image = preprocess_image(source_image_path, device)
mask = preprocess_mask(mask_path, device)

image = pipeline(
    prompt=prompt, 
    image=source_image,
    mask_image=mask,
    height=1024,
    width=1024,
    AAS=True, # enable AAS
    strength=0.8, # inpainting strength
    rm_guidance_scale=9, # removal guidance scale
    ss_steps = 9, # similarity suppression steps
    ss_scale = 0.3, # similarity suppression scale
    AAS_start_step=0, # AAS start step
    AAS_start_layer=34, # AAS start layer
    AAS_end_layer=70, # AAS end layer
    num_inference_steps=50, # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
    generator=generator,
    guidance_scale=1,
).images[0]

image.save('./removed_img_torch.png')
print("Object removal completed")

# Convert the AAS-modified UNet to OpenVINO IR after generation.
convert_unet_to_openvino(pipeline.unet)