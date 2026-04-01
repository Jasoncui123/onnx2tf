import os
import subprocess
import sys
import tempfile
from importlib import reload
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_TORCH_IMPORTS = """
import importlib.abc
import sys

class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(
                "blocked torch import for test",
                name="torch",
            )
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            raise ModuleNotFoundError(
                "blocked torchvision import for test",
                name="torchvision",
            )
        if fullname == "torchaudio" or fullname.startswith("torchaudio."):
            raise ModuleNotFoundError(
                "blocked torchaudio import for test",
                name="torchaudio",
            )
        return None

sys.meta_path.insert(0, _BlockTorch())
"""


def _run_python(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _make_add_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, path)


def test_imports_do_not_load_pytorch_when_blocked() -> None:
    template = BLOCK_TORCH_IMPORTS + """
import sys
import {module_name}
print('torch' in sys.modules)
"""
    modules = [
        "onnx2tf",
        "onnx2tf.onnx2tf",
        "onnx2tf.tflite_builder.__init__",
        "onnx2tf.utils.pytorch_bulk_runner",
        "onnx2tf.tflite_builder.pytorch_accuracy_evaluator",
        "onnx2tf.tflite_builder.pytorch_package_runtime",
    ]
    for module_name in modules:
        result = _run_python(template.format(module_name=module_name))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False", module_name


def test_pytorch_bulk_help_works_without_pytorch() -> None:
    result = _run_python(
        BLOCK_TORCH_IMPORTS
        + "import sys, onnx2tf.utils.pytorch_bulk_runner as m; sys.argv=['onnx2tf-pytorch-bulk', '--help']; m.main()"
    )
    assert result.returncode == 0
    assert "--list_path" in result.stdout


def test_flatbuffer_direct_pytorch_export_fails_fast_without_pytorch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TORCH_IMPORTS
            + """
import sys
import onnx2tf

try:
    onnx2tf.convert(
        input_onnx_file_path=sys.argv[1],
        output_folder_path=sys.argv[2],
        tflite_backend='flatbuffer_direct',
        flatbuffer_direct_output_pytorch=True,
        disable_strict_mode=True,
        verbosity='error',
    )
except Exception as ex:
    print(type(ex).__name__)
    print(str(ex))
""",
            str(model_path),
            str(output_dir),
        )
        assert result.returncode == 0, result.stderr
        assert "OptionalPyTorchDependencyError" in result.stdout
        assert "flatbuffer_direct PyTorch package export" in result.stdout
        assert 'uv pip install -U "onnx2tf[torch]"' in result.stdout
        assert 'onnx2tf[torch]' in result.stdout


def test_main_torchscript_export_fails_fast_without_pytorch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TORCH_IMPORTS
            + """
import sys
import onnx2tf

sys.argv = [
    'onnx2tf',
    '-i', sys.argv[1],
    '-o', sys.argv[2],
    '--tflite_backend', 'flatbuffer_direct',
    '--flatbuffer_direct_output_torchscript',
]
onnx2tf.main()
""",
            str(model_path),
            str(output_dir),
        )
        assert result.returncode == 1
        assert "flatbuffer_direct TorchScript export" in result.stdout
        assert 'uv pip install -U "onnx2tf[torch]"' in result.stdout
        assert 'onnx2tf[torch]' in result.stdout
        assert "Traceback" not in result.stderr


def test_pytorch_bulk_runner_fails_fast_without_pytorch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        list_path = Path(tmpdir) / "list.txt"
        output_dir = Path(tmpdir) / "out"
        list_path.write_text("", encoding="utf-8")
        result = _run_python(
            BLOCK_TORCH_IMPORTS
            + """
import sys
import onnx2tf.utils.pytorch_bulk_runner as m

sys.argv = [
    'onnx2tf-pytorch-bulk',
    '-l', sys.argv[1],
    '-o', sys.argv[2],
]
m.main()
""",
            str(list_path),
            str(output_dir),
        )
        assert result.returncode == 1
        assert "PyTorch bulk verification" in result.stdout
        assert 'uv pip install -U "onnx2tf[torch]"' in result.stdout
        assert 'onnx2tf[torch]' in result.stdout
        assert "Traceback" not in result.stderr


def test_pytorch_evaluator_call_fails_fast_without_pytorch() -> None:
    result = _run_python(
        BLOCK_TORCH_IMPORTS
        + """
import onnx
from onnx import TensorProto, helper
from onnx2tf.tflite_builder.pytorch_accuracy_evaluator import evaluate_tflite_pytorch_package_outputs

graph = helper.make_graph(
    [helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")],
    "add_graph",
    [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3]),
    ],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

try:
    evaluate_tflite_pytorch_package_outputs(
        tflite_path="dummy.tflite",
        package_dir="dummy_pkg",
        output_report_path="dummy.json",
    )
except Exception as ex:
    print(type(ex).__name__)
    print(str(ex))
""",
        )
    assert result.returncode == 0, result.stderr
    assert "OptionalPyTorchDependencyError" in result.stdout
    assert "TFLite/PyTorch validation" in result.stdout
    assert 'uv pip install -U "onnx2tf[torch]"' in result.stdout
    assert 'onnx2tf[torch]' in result.stdout


def test_pytorch_runtime_call_fails_fast_without_pytorch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        package_dir.mkdir()
        result = _run_python(
            BLOCK_TORCH_IMPORTS
            + """
import sys
from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package

try:
    load_generated_model_package(package_dir=sys.argv[1])
except Exception as ex:
    print(type(ex).__name__)
    print(str(ex))
""",
            str(package_dir),
        )
        assert result.returncode == 0, result.stderr
        assert "OptionalPyTorchDependencyError" in result.stdout
        assert "PyTorch package runtime" in result.stdout
        assert 'uv pip install -U "onnx2tf[torch]"' in result.stdout
        assert 'onnx2tf[torch]' in result.stdout


def test_require_torch_reports_foreign_runtime_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    import onnx2tf.utils.torch_optional as torch_optional

    reload(torch_optional)
    monkeypatch.setenv("PYTHONPATH", "/usr/local/lib/python3.10/dist-packages:")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/home/example/.local/lib/python3.10/site-packages/torch/lib",
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError(
                "/home/example/.local/lib/python3.10/site-packages/torch/lib/"
                "libtorch_python.so: undefined symbol: _PyCode_GetExtra"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(torch_optional.OptionalPyTorchDependencyError) as ex:
        torch_optional.require_torch("flatbuffer_direct TorchScript export")

    message = str(ex.value)
    assert "flatbuffer_direct TorchScript export requires a working torch runtime" in message
    assert "foreign torch runtime from another Python environment" in message
    assert "/home/example/.local/lib/python3.10/site-packages/torch/lib/libtorch_python.so" in message
    assert "Unset PYTHONPATH" in message
