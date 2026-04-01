from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


class OptionalTensorFlowDependencyError(RuntimeError):
    """Raised when a TensorFlow-backed feature is requested without TensorFlow."""


@dataclass(frozen=True)
class TensorFlowModules:
    tf: Any
    tf_keras: Any
    wrapper_function_type: Any


_TF_MODULES: Optional[TensorFlowModules] = None


def _should_suppress_tf_startup_stderr() -> bool:
    raw = str(
        os.environ.get(
            "ONNX2TF_SUPPRESS_TF_STARTUP_STDERR",
            "1",
        )
    ).strip().lower()
    return raw not in {"0", "false", "off", "no"}


@contextlib.contextmanager
def _suppress_stderr_fd_for_tf_startup():
    if not _should_suppress_tf_startup_stderr():
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = None
    devnull_fd = None
    try:
        saved_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        try:
            if saved_fd is not None:
                os.dup2(saved_fd, stderr_fd)
        finally:
            if saved_fd is not None:
                os.close(saved_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)


def _build_missing_dependency_message(feature: str) -> str:
    normalized_feature = str(feature).strip() or "TensorFlow-backed onnx2tf feature"
    return (
        f"{normalized_feature} requires optional dependencies: tensorflow, tf_keras. "
        'Install them with: uv pip install -U "onnx2tf[tensorflow]"'
    )


def require_tensorflow(feature: str) -> TensorFlowModules:
    global _TF_MODULES
    if _TF_MODULES is not None:
        return _TF_MODULES

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    try:
        with _suppress_stderr_fd_for_tf_startup():
            import tensorflow as tf
            from tensorflow.python.saved_model.load import _WrapperFunction
            import tf_keras
    except ModuleNotFoundError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name in {"tensorflow", "tf_keras"}:
            raise OptionalTensorFlowDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise
    except ImportError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name in {"tensorflow", "tf_keras"}:
            raise OptionalTensorFlowDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise

    tf.random.set_seed(0)
    tf_keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.FATAL)
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)

    _TF_MODULES = TensorFlowModules(
        tf=tf,
        tf_keras=tf_keras,
        wrapper_function_type=_WrapperFunction,
    )
    return _TF_MODULES
