# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from conftest import SAMPLES_PY_DIR
from test_utils import run_sample


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