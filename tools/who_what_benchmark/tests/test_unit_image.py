# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_parse_args_accepts_attentive_eraser_flag(monkeypatch):
    from whowhatbench.wwb import parse_args

    monkeypatch.setattr(sys, "argv", ["wwb", "--attentive-eraser"])

    assert parse_args().attentive_eraser is True


def test_attentive_eraser_requires_image_inpainting_model_type(monkeypatch):
    from whowhatbench.wwb import check_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["wwb", "--base-model", "sdxl-base", "--attentive-eraser", "--hf"],
    )

    with pytest.raises(ValueError, match="--model-type image-inpainting"):
        check_args(parse_args())


def test_attentive_eraser_base_generation_requires_hf(monkeypatch):
    from whowhatbench.wwb import check_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wwb",
            "--base-model",
            "sdxl-base",
            "--gt-data",
            "reference.csv",
            "--model-type",
            "image-inpainting",
            "--attentive-eraser",
        ],
    )

    with pytest.raises(ValueError, match="reference generation requires --hf"):
        check_args(parse_args())


def test_attentive_eraser_target_generation_requires_genai(monkeypatch):
    from whowhatbench.wwb import check_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wwb",
            "--gt-data",
            "reference.csv",
            "--target-model",
            "sdxl-openvino",
            "--model-type",
            "image-inpainting",
            "--attentive-eraser",
        ],
    )

    with pytest.raises(ValueError, match="target generation requires --genai"):
        check_args(parse_args())


@pytest.mark.parametrize("image_size", [64, 512, 2048])
def test_attentive_eraser_rejects_unsupported_image_size(monkeypatch, image_size):
    from whowhatbench.wwb import check_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wwb",
            "--base-model",
            "sdxl-base",
            "--gt-data",
            "reference.csv",
            "--model-type",
            "image-inpainting",
            "--attentive-eraser",
            "--hf",
            "--image-size",
            str(image_size),
        ],
    )

    with pytest.raises(ValueError, match="requires --image-size 1024"):
        check_args(parse_args())


@pytest.mark.parametrize("image_size_args", [[], ["--image-size", "1024"]])
def test_attentive_eraser_accepts_default_or_1024_image_size(monkeypatch, image_size_args):
    from whowhatbench.wwb import check_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wwb",
            "--base-model",
            "sdxl-base",
            "--gt-data",
            "reference.csv",
            "--model-type",
            "image-inpainting",
            "--attentive-eraser",
            "--hf",
            *image_size_args,
        ],
    )

    check_args(parse_args())


@pytest.mark.parametrize(
    ("backend_flag", "expected_generation_fn"),
    [
        ("--hf", "diffusers_gen_attentive_eraser"),
        ("--genai", "genai_gen_attentive_eraser"),
    ],
)
def test_create_evaluator_selects_attentive_eraser_generation_adapter(
    monkeypatch, backend_flag, expected_generation_fn
):
    from whowhatbench import wwb

    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(wwb.EVALUATOR_REGISTRY, "image-inpainting", FakeEvaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wwb",
            "--gt-data",
            "reference.csv",
            "--model-type",
            "image-inpainting",
            "--attentive-eraser",
            backend_flag,
        ],
    )

    wwb.create_evaluator(object(), wwb.parse_args())

    assert captured["gen_image_fn"] is getattr(wwb, expected_generation_fn)


def test_text2image_genai_npu_reshapes_before_compile(monkeypatch):
    """Test that NPU text-to-image pipeline reshapes to static shapes before compilation."""
    from whowhatbench import model_loaders

    calls = []

    class FakePipeline:
        def __init__(self, model_dir):
            calls.append(("init", model_dir))

        def get_generation_config(self):
            calls.append(("get_generation_config",))
            return SimpleNamespace(guidance_scale=7.5)

        def reshape(self, **kwargs):
            calls.append(("reshape", kwargs))

        def compile(self, device, **properties):
            calls.append(("compile", device, properties))

    class FakeGenAIModelWrapper:
        def __init__(self, model, model_dir, model_type):
            self.model = model
            self.model_dir = model_dir
            self.model_type = model_type

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(Text2ImagePipeline=FakePipeline),
    )
    monkeypatch.setattr(model_loaders, "GenAIModelWrapper", FakeGenAIModelWrapper)

    wrapper = model_loaders.load_text2image_genai_pipeline(
        "model_dir",
        device="NPU",
        ov_config={"CACHE_DIR": "cache"},
        image_size=512,
    )

    assert wrapper.model_type == "text-to-image"
    assert calls == [
        ("init", "model_dir"),
        ("get_generation_config",),
        (
            "reshape",
            {"num_images_per_prompt": 1, "height": 512, "width": 512, "guidance_scale": 7.5},
        ),
        ("compile", "NPU", {"CACHE_DIR": "cache"}),
    ]


def test_text2image_genai_npu_requires_positive_image_size(monkeypatch):
    """Test that NPU text-to-image pipeline requires positive image-size."""
    from whowhatbench import model_loaders

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(Text2ImagePipeline=object))

    with pytest.raises(ValueError, match="positive --image-size"):
        model_loaders.load_text2image_genai_pipeline("model_dir", device="NPU", ov_config={})


def test_genai_gen_image_uses_model_adapter_config():
    """Test that genai_gen_image forwards model adapter_config to generate()."""
    from whowhatbench.wwb import genai_gen_image

    calls = []

    class FakeModel:
        resolution = (512, 512)
        adapter_config = object()

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(data=np.zeros((1, 1, 1, 3), dtype=np.uint8))

    genai_gen_image(FakeModel(), "prompt", 4)

    assert calls[0][1]["adapters"] is FakeModel.adapter_config


def test_load_inpainting_model_uses_official_attentive_eraser_pipeline(monkeypatch):
    from whowhatbench import model_loaders

    calls = []
    model = object()
    scheduler = object()

    class FakeDiffusionPipeline:
        @staticmethod
        def load_config(model_id):
            assert model_id == "sdxl-base"
            return {"_class_name": "StableDiffusionXLPipeline"}

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return model

    class FakeDDIMScheduler:
        @staticmethod
        def from_pretrained(model_id, subfolder):
            assert model_id == "sdxl-base"
            assert subfolder == "scheduler"
            return scheduler

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            DDIMScheduler=FakeDDIMScheduler,
            DiffusionPipeline=FakeDiffusionPipeline,
        ),
    )
    monkeypatch.setattr(model_loaders, "disable_diffusers_model_progress_bar", lambda _: None)

    loaded_model = model_loaders.load_inpainting_model(
        "sdxl-base",
        use_hf=True,
        attentive_eraser=True,
    )

    assert loaded_model is model
    assert calls == [
        (
            "sdxl-base",
            {
                "custom_pipeline": "pipeline_stable_diffusion_xl_attentive_eraser",
                "custom_revision": "0.39.0",
                "scheduler": scheduler,
                "torch_dtype": model_loaders.torch.float32,
                "use_safetensors": True,
                "variant": "fp16",
            },
        )
    ]


def test_load_attentive_eraser_rejects_non_sdxl_model(monkeypatch):
    from whowhatbench import model_loaders

    class FakeDiffusionPipeline:
        @staticmethod
        def load_config(model_id):
            assert model_id == "sd15-base"
            return {"_class_name": "StableDiffusionPipeline"}

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline),
    )

    with pytest.raises(ValueError, match="requires an SDXL Base model"):
        model_loaders.load_inpainting_model(
            "sd15-base",
            use_hf=True,
            attentive_eraser=True,
        )


def test_load_attentive_eraser_requires_supported_backend(monkeypatch):
    from whowhatbench import model_loaders

    class FakeDiffusionPipeline:
        @staticmethod
        def load_config(model_id):
            assert model_id == "sdxl-base"
            return {"_class_name": "StableDiffusionXLPipeline"}

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline),
    )

    with pytest.raises(ValueError, match="requires the HF or GenAI backend"):
        model_loaders.load_inpainting_model(
            "sdxl-base",
            attentive_eraser=True,
        )


def test_load_inpainting_genai_pipeline_enables_attentive_eraser(monkeypatch):
    from whowhatbench import model_loaders

    calls = []
    pipeline = object()
    attentive_eraser_mode = object()

    def create_pipeline(model_dir, device, **kwargs):
        calls.append((model_dir, device, kwargs))
        return pipeline

    class FakeGenAIModelWrapper:
        def __init__(self, model, model_dir, model_type):
            self.model = model
            self.model_dir = model_dir
            self.model_type = model_type

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(
            InpaintingMode=SimpleNamespace(ATTENTIVE_ERASER=attentive_eraser_mode),
            InpaintingPipeline=create_pipeline,
        ),
    )
    monkeypatch.setattr(model_loaders, "GenAIModelWrapper", FakeGenAIModelWrapper)

    wrapper = model_loaders.load_inpainting_genai_pipeline(
        "model_dir",
        device="GPU",
        ov_config={"CACHE_DIR": "cache"},
        attentive_eraser=True,
    )

    assert wrapper.model is pipeline
    assert calls == [
        (
            "model_dir",
            "GPU",
            {
                "CACHE_DIR": "cache",
                "inpainting_mode": attentive_eraser_mode,
            },
        )
    ]


def test_diffusers_attentive_eraser_generation_uses_fixed_sdxl_parameters():
    from whowhatbench import model_loaders
    from whowhatbench.wwb import diffusers_gen_attentive_eraser

    calls = []
    output_image = object()
    source_image = np.full((8, 8, 3), 255, dtype=np.uint8)
    source_mask = np.zeros((8, 8, 3), dtype=np.uint8)
    source_mask[2:6, 2:6] = 255
    generator = object()

    class FakeModel:
        unet = SimpleNamespace(device="cpu", dtype=model_loaders.torch.float32)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[output_image])

    result = diffusers_gen_attentive_eraser(
        FakeModel(),
        "caption used only as metadata",
        image=source_image,
        mask=source_mask,
        num_inference_steps=12,
        generator=generator,
    )

    assert result is output_image
    image = calls[0].pop("image")
    mask = calls[0].pop("mask_image")
    assert tuple(image.shape) == (1, 3, 1024, 1024)
    assert tuple(mask.shape) == (1, 1, 1024, 1024)
    assert image.min().item() == 1.0
    assert image.max().item() == 1.0
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert calls == [{
        "prompt": "",
        "num_inference_steps": 12,
        "generator": generator,
        "output_type": "pil",
        "AAS": True,
        "strength": 0.8,
        "guidance_scale": 1.0,
        "rm_guidance_scale": 9.0,
        "ss_steps": 9,
        "ss_scale": 0.3,
        "AAS_start_step": 0,
        "AAS_start_layer": 34,
        "AAS_end_layer": 70,
    }]


def test_genai_attentive_eraser_generation_uses_fixed_sdxl_parameters(monkeypatch):
    from whowhatbench import wwb

    calls = []
    generator = object()
    generation_config = SimpleNamespace()

    class FakeAttentiveEraserConfig:
        pass

    class FakeModel:
        def get_generation_config(self):
            return generation_config

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(data=np.zeros((1, 1, 1, 3), dtype=np.uint8))

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(AttentiveEraserConfig=FakeAttentiveEraserConfig),
    )

    wwb.genai_gen_attentive_eraser(
        FakeModel(),
        "caption used only as metadata",
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        mask=np.zeros((1, 1, 3), dtype=np.uint8),
        num_inference_steps=12,
        generator=generator,
    )

    attentive_eraser = generation_config.attentive_eraser
    assert generation_config.strength == 0.8
    assert generation_config.guidance_scale == 1.0
    assert generation_config.num_inference_steps == 12
    assert attentive_eraser.rm_guidance_scale == 9.0
    assert attentive_eraser.ss_steps == 9
    assert attentive_eraser.start_step == 0
    assert attentive_eraser.ss_scale == 0.3
    assert attentive_eraser.mask_blur_kernel == 77
    assert calls[0][0] == ""
    assert calls[0][1]["generation_config"] is generation_config
    assert calls[0][1]["generator"] is generator
