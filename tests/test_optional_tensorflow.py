import os
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_TF_IMPORTS = """
import importlib.abc
import sys

class _BlockTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise ModuleNotFoundError(
                "blocked tensorflow import for test",
                name="tensorflow",
            )
        if fullname == "tf_keras" or fullname.startswith("tf_keras."):
            raise ModuleNotFoundError(
                "blocked tf_keras import for test",
                name="tf_keras",
            )
        return None

sys.meta_path.insert(0, _BlockTensorFlow())
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


def test_imports_do_not_load_tensorflow_when_blocked() -> None:
    template = BLOCK_TF_IMPORTS + """
import sys
import {module_name}
print('tensorflow' in sys.modules, 'tf_keras' in sys.modules)
"""
    modules = [
        "onnx2tf",
        "onnx2tf.utils.flatbuffer_direct_bulk_runner",
        "onnx2tf.utils.flatbuffer_direct_op_error_report",
        "onnx2tf.tflite_builder.accuracy_evaluator",
        "onnx2tf.onnx2tf",
    ]
    for module_name in modules:
        result = _run_python(template.format(module_name=module_name))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False False", module_name


def test_python_m_help_works_without_tensorflow() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BLOCK_TF_IMPORTS
            + "import runpy, sys; sys.argv=['onnx2tf','--help']; runpy.run_module('onnx2tf', run_name='__main__')",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert "--tflite_backend" in result.stdout


def test_main_help_works_without_tensorflow() -> None:
    result = _run_python(
        BLOCK_TF_IMPORTS
        + "import sys, onnx2tf; sys.argv=['onnx2tf', '--help']; onnx2tf.main()"
    )
    assert result.returncode == 0
    assert "--tflite_backend" in result.stdout


def test_flatbuffer_direct_convert_succeeds_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import os
import sys
import onnx2tf

model_path = sys.argv[1]
output_dir = sys.argv[2]
onnx2tf.convert(
    input_onnx_file_path=model_path,
    output_folder_path=output_dir,
    tflite_backend='flatbuffer_direct',
    disable_strict_mode=True,
    verbosity='error',
)
print(os.path.exists(os.path.join(output_dir, 'add_float32.tflite')))
print('tensorflow' in sys.modules, 'tf_keras' in sys.modules)
""",
            str(model_path),
            str(output_dir),
        )
        assert result.returncode == 0, result.stderr
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert lines[-2] == "True"
        assert lines[-1] == "False False"


def test_flatbuffer_direct_cotof_succeeds_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import os
import sys
import onnx2tf

model_path = sys.argv[1]
output_dir = sys.argv[2]
onnx2tf.convert(
    input_onnx_file_path=model_path,
    output_folder_path=output_dir,
    tflite_backend='flatbuffer_direct',
    check_onnx_tf_outputs_elementwise_close_full=True,
    disable_strict_mode=True,
    verbosity='error',
)
print(os.path.exists(os.path.join(output_dir, 'add_accuracy_report.json')))
print(os.path.exists(os.path.join(output_dir, 'add_saved_model_validation_report.json')))
print('tensorflow' in sys.modules, 'tf_keras' in sys.modules)
""",
            str(model_path),
            str(output_dir),
        )
        assert result.returncode == 0, result.stderr
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert lines[-3] == "True"
        assert lines[-2] == "False"
        assert lines[-1] == "False False"


def test_tf_converter_fails_fast_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import sys
import onnx2tf

try:
    onnx2tf.convert(
        input_onnx_file_path=sys.argv[1],
        output_folder_path=sys.argv[2],
        tflite_backend='tf_converter',
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
        assert "OptionalTensorFlowDependencyError" in result.stdout
        assert 'tflite_backend="tf_converter"' in result.stdout
        assert 'uv pip install -U "onnx2tf[tensorflow]"' in result.stdout
        assert 'onnx2tf[tensorflow]' in result.stdout


def test_main_tf_converter_fails_fast_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import sys
import onnx2tf

sys.argv = [
    'onnx2tf',
    '-i', sys.argv[1],
    '-o', sys.argv[2],
    '--tflite_backend', 'tf_converter',
]
onnx2tf.main()
""",
            str(model_path),
            str(output_dir),
        )
        assert result.returncode == 1
        assert 'tflite_backend="tf_converter"' in result.stdout
        assert 'uv pip install -U "onnx2tf[tensorflow]"' in result.stdout
        assert 'onnx2tf[tensorflow]' in result.stdout
        assert "Traceback" not in result.stderr


def test_flatbuffer_direct_h5_fails_fast_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import sys
import onnx2tf

try:
    onnx2tf.convert(
        input_onnx_file_path=sys.argv[1],
        output_folder_path=sys.argv[2],
        tflite_backend='flatbuffer_direct',
        output_h5=True,
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
        assert "OptionalTensorFlowDependencyError" in result.stdout
        assert "flatbuffer_direct H5 export" in result.stdout
        assert 'uv pip install -U "onnx2tf[tensorflow]"' in result.stdout
        assert 'onnx2tf[tensorflow]' in result.stdout


def test_flatbuffer_direct_saved_model_with_cotof_fails_fast_without_tensorflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "add.onnx"
        output_dir = Path(tmpdir) / "out"
        _make_add_model(model_path)
        result = _run_python(
            BLOCK_TF_IMPORTS
            + """
import sys
import onnx2tf

try:
    onnx2tf.convert(
        input_onnx_file_path=sys.argv[1],
        output_folder_path=sys.argv[2],
        tflite_backend='flatbuffer_direct',
        flatbuffer_direct_output_saved_model=True,
        check_onnx_tf_outputs_elementwise_close_full=True,
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
        assert "OptionalTensorFlowDependencyError" in result.stdout
        assert "flatbuffer_direct SavedModel export" in result.stdout
        assert 'uv pip install -U "onnx2tf[tensorflow]"' in result.stdout
        assert 'onnx2tf[tensorflow]' in result.stdout
