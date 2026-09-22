# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import SAMPLES_PY_DIR
from test_utils import run_sample


def load_attentive_eraser_sample():
    sample_path = SAMPLES_PY_DIR / "image_generation/attentive_eraser_pipeline.py"
    spec = importlib.util.spec_from_file_location("attentive_eraser_pipeline", sample_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAttentiveEraser:
    @pytest.mark.samples
    def test_python_sample_help(self):
        result = run_sample(
            [
                sys.executable,
                SAMPLES_PY_DIR / "image_generation/attentive_eraser_pipeline.py",
                "--help",
            ]
        )

        assert "MODEL_DIR IMAGE MASK_IMAGE [DEVICE] [SEED]" in result.stdout
        assert "--height HEIGHT" in result.stdout
        assert "--width WIDTH" in result.stdout
        assert "--pipeline-shape {dynamic,static}" in result.stdout

    @pytest.mark.samples
    def test_python_sample_static_shape_requires_explicit_dimensions(self):
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_sample(
                [
                    sys.executable,
                    SAMPLES_PY_DIR / "image_generation/attentive_eraser_pipeline.py",
                    "model",
                    "image.png",
                    "mask.png",
                    "--pipeline-shape",
                    "static",
                ]
            )

        assert "static pipeline shape requires --height and --width" in error.value.stdout

    @pytest.mark.samples
    @pytest.mark.parametrize(
        ("pipeline_shape", "expected_calls"),
        [
            (
                "static",
                [
                    "set_generation_config",
                    ("reshape", 1, 512, 768, 1.0),
                    ("compile", "GPU"),
                    "generate",
                ],
            ),
            (
                "dynamic",
                ["set_generation_config", ("compile", "GPU"), "generate"],
            ),
        ],
    )
    def test_python_sample_configures_shape_before_compile(
        self, monkeypatch, pipeline_shape, expected_calls
    ):
        sample = load_attentive_eraser_sample()
        calls = []
        constructor_args = []

        class FakePipeline:
            def __init__(self, *args):
                constructor_args.append(args)
                self.config = SimpleNamespace(guidance_scale=1.0)

            def get_generation_config(self):
                return self.config

            def set_generation_config(self, config):
                calls.append("set_generation_config")

            def reshape(self, num_images, height, width, guidance_scale):
                calls.append(("reshape", num_images, height, width, guidance_scale))

            def compile(self, device):
                calls.append(("compile", device))

            def generate(self, prompt, image, mask, callback):
                calls.append("generate")
                return SimpleNamespace(data=[[[0, 0, 0]]])

        monkeypatch.setattr(sample.openvino_genai, "InpaintingPipeline", FakePipeline)
        monkeypatch.setattr(sample, "read_image", lambda path: object())
        monkeypatch.setattr(sample.Image, "fromarray", lambda data: SimpleNamespace(save=lambda path: None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "attentive_eraser_pipeline.py",
                "model",
                "image.png",
                "mask.png",
                "GPU",
                "123",
                "--height",
                "512",
                "--width",
                "768",
                "--pipeline-shape",
                pipeline_shape,
            ],
        )

        sample.main()

        assert constructor_args == [
            ("model", sample.openvino_genai.InpaintingMode.ATTENTIVE_ERASER)
        ]
        assert calls == expected_calls

    @pytest.mark.samples
    def test_python_sample_uses_rgb_loader_for_image_and_mask(self, tmp_path):
        sample = load_attentive_eraser_sample()
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        Image.new("RGB", (2, 1), "white").save(image_path)
        Image.new("L", (2, 1), 255).save(mask_path)

        image = sample.read_image(image_path)
        mask = sample.read_image(mask_path)

        assert image.shape == [1, 1, 2, 3]
        assert mask.shape == [1, 1, 2, 3]
        assert not hasattr(sample, "read_mask")
