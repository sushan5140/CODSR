import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from my_utils import wavelet_color_fix as color_fix

CLI = Path(__file__).resolve().parents[1] / "test_codsr.py"


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    def run(source, upscale, align_method, process_size=512):
        output_dir = tmp_path / "output"
        model_sizes, alignment_calls, output_resizes = [], [], []
        saving_output = False

        class Passthrough:
            def __init__(self, args):
                pass

            def cuda(self):
                return self

            def __call__(self, image, *, prompt):
                nonlocal saving_output
                saving_output = True
                model_sizes.append(tuple(reversed(image.shape[-2:])))
                return image

        class CaptionModel:
            def eval(self):
                return self

            def to(self, *args, **kwargs):
                return self

        def record_alignment(name, align):
            def wrapped(*, target, source):
                alignment_calls.append((name, target.size, source.size))
                return align(target, source)

            return wrapped

        resize = Image.Image.resize
        save = Image.Image.save
        to = torch.Tensor.to

        def record_output_resize(image, size, *args, **kwargs):
            if saving_output:
                output_resizes.append((image.size, size))
            return resize(image, size, *args, **kwargs)

        def finish_output(image, *args, **kwargs):
            nonlocal saving_output
            saving_output = False
            return save(image, *args, **kwargs)

        def use_cpu(tensor, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0] == "cuda":
                args = ("cpu", *args[1:])
            return to(tensor, *args, **kwargs)

        codsr = ModuleType("codsr")
        codsr.CODSR_test = Passthrough
        ram = ModuleType("ram")
        ram.inference_ram = lambda *_: ["fixture caption"]
        ram_lora = ModuleType("ram.models.ram_lora")
        ram_lora.ram = lambda **_: CaptionModel()

        with monkeypatch.context() as local, patch.dict(os.environ):
            local.setitem(sys.modules, "codsr", codsr)
            local.setitem(sys.modules, "ram", ram)
            local.setitem(sys.modules, "ram.models.ram_lora", ram_lora)
            local.setattr(sys, "path", sys.path.copy())
            local.setattr(sys, "argv", [
                str(CLI), "-i", str(source), "-o", str(output_dir),
                "--upscale", str(upscale), "--align_method", align_method,
                "--process_size", str(process_size), "--mixed_precision", "fp32",
            ])
            local.setattr(torch.Tensor, "to", use_cpu)
            local.setattr(torch, "set_num_threads", lambda _: None)
            local.setattr(torch.cuda, "current_device", lambda: 0)
            local.setattr(torch.cuda, "empty_cache", lambda: None)
            local.setattr(Image.Image, "resize", record_output_resize)
            local.setattr(Image.Image, "save", finish_output)
            for name in ("adain", "wavelet"):
                attribute = f"{name}_color_fix"
                local.setattr(color_fix, attribute, record_alignment(name, getattr(color_fix, attribute)))
            runpy.run_path(str(CLI), run_name="__main__")

        return output_dir, model_sizes, alignment_calls, output_resizes

    return run


@pytest.mark.parametrize("size,upscale,process_size,internal_size", [
    ((20, 30), 4, 512, (512, 768)),
    ((26, 26), 4, 512, (512, 512)),
    ((120, 136), 4, 512, (512, 576)),
    ((129, 131), 4, 512, (512, 520)),
    ((270, 257), 2, 512, (536, 512)),
    ((132, 130), 4, 512, (528, 520)),
    ((128, 128), 4, 512, (512, 512)),
    ((17, 21), 3, 48, (48, 56)),
])
def test_saved_dimensions_and_resize_calls(tmp_path, run_cli, size, upscale, process_size, internal_size):
    source = tmp_path / "input.png"
    Image.new("RGB", size, (0, 255, 255)).save(source)
    source_bytes = source.read_bytes()
    output, model_sizes, alignment_calls, resizes = run_cli(source, upscale, "nofix", process_size)
    target_size = tuple(side * upscale for side in size)

    assert source.read_bytes() == source_bytes
    assert model_sizes == [internal_size]
    assert alignment_calls == []
    with Image.open(output / source.name) as image:
        assert image.size == target_size
        assert image.mode == "RGB"
        assert image.format == "PNG"
        assert image.getextrema() == ((0, 0), (255, 255), (255, 255))
    assert resizes == ([] if internal_size == target_size else [(internal_size, target_size)])


@pytest.mark.parametrize("align_method", ["nofix", "adain", "wavelet"])
def test_color_alignment_and_multiple_inputs(tmp_path, run_cli, align_method):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGBA", (35, 37), (255, 0, 255, 0)).save(source / "rounded.png")
    Image.new("L", (32, 40), 255).save(source / "aligned.png")
    original = {image.name: image.read_bytes() for image in source.iterdir()}

    output, model_sizes, alignment_calls, resizes = run_cli(source, 2, align_method, process_size=64)
    assert {image.name: image.read_bytes() for image in source.iterdir()} == original
    assert {image.name for image in output.glob("*.png")} == original.keys()
    assert model_sizes == [(64, 80), (64, 72)]
    expected_calls = [(align_method, size, size) for size in model_sizes]
    assert alignment_calls == ([] if align_method == "nofix" else expected_calls)

    with Image.open(output / "aligned.png") as image:
        assert (image.size, image.mode, image.format) == ((64, 80), "RGB", "PNG")
        assert image.getextrema() == ((255, 255),) * 3
    with Image.open(output / "rounded.png") as image:
        assert (image.size, image.mode, image.format) == ((70, 74), "RGB", "PNG")
        assert image.getextrema() == ((255, 255), (0, 0), (255, 255))
    assert resizes == [((64, 72), (70, 74))]
