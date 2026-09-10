import importlib.util
from pathlib import Path

import openvino as ov
import torch


def load_conversion_sample():
    sample_path = (
        Path(__file__).parents[3]
        / "samples"
        / "cpp"
        / "image_generation"
        / "atten_eraser_torch.py"
    )
    spec = importlib.util.spec_from_file_location("atten_eraser_torch", sample_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_unet_parity_is_non_exporting_mode():
    sample = load_conversion_sample()

    args = sample.parse_args(["--model_type", "SD2", "--validate-unet-parity"])

    assert args.validate_unet_parity
    assert not args.convert_unet
    assert not args.export_model_only


def test_mask_blur_kernel_matches_model_family():
    sample = load_conversion_sample()

    assert sample.MODEL_CONFIG["SD15"]["mask_blur_kernel"] == 7
    assert sample.MODEL_CONFIG["SD2"]["mask_blur_kernel"] == 7
    assert sample.MODEL_CONFIG["SDXL"]["mask_blur_kernel"] == 77


def test_conversion_uses_a_torch_dtype_supported_by_the_device():
    sample = load_conversion_sample()

    expected_dtype = torch.float16 if sample.device.type == "cuda" else torch.float32

    assert sample.dtype == expected_dtype


def test_custom_pipeline_configuration_matches_model_family():
    sample = load_conversion_sample()

    assert Path(sample.MODEL_CONFIG["SD15"]["custom_pipeline"]).is_file()
    assert sample.MODEL_CONFIG["SDXL"]["custom_pipeline"] == "pipeline_stable_diffusion_xl_attentive_eraser"
    assert sample.MODEL_CONFIG["SDXL"]["custom_revision"] == "0.39.0"
    local_sdxl_pipeline = (
        Path(__file__).parents[3]
        / "samples"
        / "cpp"
        / "image_generation"
        / "atten_eraser_pipeline"
        / "pipeline_stable_diffusion_xl_attentive_eraser.py"
    )
    assert not local_sdxl_pipeline.exists()


def test_stateful_unet_wrapper_is_traced_without_consistency_replay():
    sample = load_conversion_sample()

    class StatefulWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, model_sample, timestep):
            self.calls += 1
            return model_sample + timestep.to(model_sample.dtype)

    wrapper = StatefulWrapper()
    example_input = {
        "model_sample": torch.ones(1),
        "timestep": torch.tensor(1, dtype=torch.int64),
    }

    traced_model = sample.trace_unet_for_export(wrapper, example_input)
    converted_model = ov.convert_model(traced_model)

    assert wrapper.calls == 1
    assert {model_input.get_any_name() for model_input in converted_model.inputs} == {"model_sample", "timestep"}


def test_single_softmax_configuration_is_not_exposed():
    sample = load_conversion_sample()

    args = sample.parse_args(["--model_type", "SD15"])

    assert not hasattr(args, "use_single_softmax_output_gating")
    image_generation_samples = Path(__file__).parents[3] / "samples" / "cpp" / "image_generation"
    for relative_path in (
        "atten_eraser_torch.py",
        "atten_eraser_pipeline/pipeline_inp.py",
    ):
        assert "use_single_softmax_output_gating" not in (image_generation_samples / relative_path).read_text()


def test_aas_mask_keeps_native_softmax_static_rank():
    sample = load_conversion_sample()

    class AASMaskedSoftmax(torch.nn.Module):
        def forward(self, mask, scores):
            key_mask = sample._downsample_aas_mask(mask, (8, 8)).reshape(1, 1, -1)
            penalty = key_mask.masked_fill(key_mask == 1, torch.finfo(scores.dtype).min)
            return (scores + penalty).softmax(dim=2)

    traced_model = torch.jit.trace(
        AASMaskedSoftmax(),
        (torch.ones(1, 1, 64, 64), torch.randn(2, 64, 64)),
        check_trace=False,
    )
    converted_model = ov.convert_model(traced_model)
    converted_model.reshape({converted_model.input(0): [1, 1, -1, -1], converted_model.input(1): [-1, -1, -1]})

    softmax_operations = [operation for operation in converted_model.get_ops() if operation.get_type_name() == "Softmax"]
    assert len(softmax_operations) == 1
    assert softmax_operations[0].input_value(0).get_partial_shape().rank.is_static
    ov.Core().compile_model(converted_model, "CPU")


def test_exported_unet_uses_fp32_data_and_int64_step_boundaries():
    sample = load_conversion_sample()
    parameters = [
        ov.opset13.parameter([2, 4, 8, 8], ov.Type.f16, name="sample"),
        ov.opset13.parameter([], ov.Type.i64, name="timestep"),
        ov.opset13.parameter([2, 77, 768], ov.Type.f16, name="encoder_hidden_states"),
        ov.opset13.parameter([2, 1280], ov.Type.f16, name="text_embeds"),
        ov.opset13.parameter([2, 6], ov.Type.f16, name="time_ids"),
        ov.opset13.parameter([1, 1, 64, 64], ov.Type.f16, name="mask.1"),
        ov.opset13.parameter([], ov.Type.i64, name="cur_step"),
        ov.opset13.parameter([], ov.Type.i64, name="ss_steps"),
    ]
    model = ov.Model([parameters[0]], parameters, "attentive_eraser_unet")

    prepared_model = sample.prepare_unet_for_export(model)
    input_types = {model_input.get_any_name(): model_input.get_element_type() for model_input in prepared_model.inputs}

    assert input_types == {
        "sample": ov.Type.f32,
        "timestep": ov.Type.i64,
        "encoder_hidden_states": ov.Type.f32,
        "text_embeds": ov.Type.f32,
        "time_ids": ov.Type.f32,
        "mask": ov.Type.f32,
        "cur_step": ov.Type.i64,
        "ss_steps": ov.Type.i64,
    }
    assert prepared_model.output(0).get_element_type() == ov.Type.f32


def test_exported_unet_preserves_cross_attention_contract(tmp_path):
    sample = load_conversion_sample()
    parameters = [
        ov.opset13.parameter([-1, 4, -1, -1], ov.Type.f32, name="sample"),
        ov.opset13.parameter([], ov.Type.i64, name="timestep"),
        ov.opset13.parameter([-1, -1, -1], ov.Type.f32, name="encoder_hidden_states"),
        ov.opset13.parameter([-1, -1], ov.Type.f32, name="text_embeds"),
        ov.opset13.parameter([-1, -1], ov.Type.f32, name="time_ids"),
        ov.opset13.parameter([-1, -1, -1, -1], ov.Type.f32, name="mask"),
        ov.opset13.parameter([], ov.Type.i64, name="cur_step"),
        ov.opset13.parameter([], ov.Type.i64, name="ss_steps"),
    ]
    model = ov.Model([parameters[0]], parameters, "attentive_eraser_unet")

    prepared_model = sample.prepare_unet_for_export(model, 2048)
    model_path = tmp_path / "openvino_model.xml"
    ov.save_model(prepared_model, model_path)
    exported_model = ov.Core().read_model(model_path)

    assert exported_model.input("encoder_hidden_states").partial_shape == ov.PartialShape([-1, 77, 2048])
    assert exported_model.input("text_embeds").partial_shape == ov.PartialShape([-1, 1280])
    assert exported_model.input("time_ids").partial_shape == ov.PartialShape([-1, 6])
    assert exported_model.input("mask").partial_shape == ov.PartialShape([1, 1, -1, -1])
    assert exported_model.input("sample").partial_shape == ov.PartialShape([2, 4, -1, -1])