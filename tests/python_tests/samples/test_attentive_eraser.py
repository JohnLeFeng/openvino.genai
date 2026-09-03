# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys

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