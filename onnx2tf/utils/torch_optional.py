from __future__ import annotations

import os
import re
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class OptionalPyTorchDependencyError(RuntimeError):
    """Raised when a PyTorch-backed feature is requested without PyTorch."""


@dataclass(frozen=True)
class PyTorchModules:
    torch: Any


_TORCH_MODULES: Optional[PyTorchModules] = None


def _build_missing_dependency_message(feature: str) -> str:
    normalized_feature = str(feature).strip() or "PyTorch-backed onnx2tf feature"
    return (
        f"{normalized_feature} requires optional dependency: torch. "
        'Install it with: uv pip install -U "onnx2tf[torch]"'
    )


def _normalize_path(path: str) -> str:
    if str(path).strip() == "":
        return ""
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except Exception:
        return os.path.normcase(os.path.abspath(str(path)))


def _current_environment_roots() -> list[str]:
    roots: list[str] = []
    for candidate in {
        sys.prefix,
        sys.exec_prefix,
        os.environ.get("VIRTUAL_ENV", ""),
    }:
        normalized = _normalize_path(str(candidate))
        if normalized and normalized not in roots:
            roots.append(normalized)
    return roots


def _is_foreign_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if normalized == "":
        return False
    env_roots = _current_environment_roots()
    return not any(
        normalized == env_root or normalized.startswith(f"{env_root}{os.sep}")
        for env_root in env_roots
    )


def _find_foreign_torch_runtime_references() -> list[str]:
    suspects: list[str] = []
    for env_name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        raw = str(os.environ.get(env_name, "") or "")
        if raw.strip() == "":
            continue
        for entry in raw.split(os.pathsep):
            if entry.strip() == "":
                continue
            lowered_entry = entry.lower()
            if "torch" not in lowered_entry and "site-packages" not in lowered_entry:
                continue
            if _is_foreign_path(entry):
                suspects.append(f"{env_name}={entry}")
    return suspects


def _extract_importerror_path(ex: ImportError) -> str:
    match = re.search(r"(/[^\s:]+)", str(ex))
    if match is None:
        return ""
    return str(match.group(1))


def _expected_torch_lib_path() -> str:
    try:
        purelib = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return ""
    candidate = purelib / "torch" / "lib"
    if not candidate.exists():
        return ""
    return str(candidate)


def _build_broken_runtime_message(feature: str, ex: ImportError) -> str:
    normalized_feature = str(feature).strip() or "PyTorch-backed onnx2tf feature"
    referenced_path = _extract_importerror_path(ex)
    foreign_references = _find_foreign_torch_runtime_references()
    lines = [
        f"{normalized_feature} requires a working torch runtime, but importing torch failed.",
    ]
    if referenced_path and _is_foreign_path(referenced_path):
        lines.append(
            "Detected a foreign torch runtime from another Python environment: "
            f"{referenced_path}"
        )
    elif foreign_references:
        lines.append(
            "Detected foreign torch-related environment entries: "
            + ", ".join(foreign_references)
        )
    else:
        lines.append(f"ImportError: {ex}")
    expected_torch_lib = _expected_torch_lib_path()
    if expected_torch_lib:
        lines.append(
            "Unset PYTHONPATH and remove incompatible torch paths from "
            f"LD_LIBRARY_PATH, then retry with this environment's torch libs: {expected_torch_lib}"
        )
    else:
        lines.append(
            "Unset PYTHONPATH and remove incompatible torch paths from LD_LIBRARY_PATH, "
            'or reinstall with: uv pip install -U "onnx2tf[torch]"'
        )
    return " ".join(lines)


def require_torch(feature: str) -> PyTorchModules:
    global _TORCH_MODULES
    if _TORCH_MODULES is not None:
        return _TORCH_MODULES

    try:
        import torch
    except ModuleNotFoundError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name == "torch":
            raise OptionalPyTorchDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise
    except ImportError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name == "torch":
            raise OptionalPyTorchDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise OptionalPyTorchDependencyError(
            _build_broken_runtime_message(feature, ex)
        ) from ex

    _TORCH_MODULES = PyTorchModules(
        torch=torch,
    )
    return _TORCH_MODULES
