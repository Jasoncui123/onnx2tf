from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional


_STATE_SCHEMA_VERSION = 1
_STATE_FILENAME = "bulk_status.json"
_SUMMARY_JSON_FILENAME = "bulk_summary.json"
_SUMMARY_MD_FILENAME = "bulk_summary.md"
_RUNS_DIRNAME = "runs"


def _create_progress_bar(
    *,
    total: int,
    initial: int = 0,
    desc: str,
):
    if int(total) <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(
        total=int(total),
        initial=int(initial),
        desc=str(desc),
        dynamic_ncols=True,
    )


def _update_progress_bar(
    progress_bar: Any,
    *,
    model_name: str,
    classification: str,
) -> None:
    if progress_bar is None:
        return
    model_label = str(model_name)
    if len(model_label) > 48:
        model_label = f"...{model_label[-45:]}"
    progress_bar.set_postfix_str(
        f"{model_label} [{classification}]",
        refresh=True,
    )
    progress_bar.update(1)


class _ProgressSpinner:
    def __init__(self, progress_bar: Any) -> None:
        self._progress_bar = progress_bar
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label = ""

    def set_context(self, *, model_name: str, status: str = "running") -> None:
        model_label = str(model_name)
        if len(model_label) > 48:
            model_label = f"...{model_label[-45:]}"
        self._label = f"{model_label} [{status}]"

    def start(self) -> None:
        self.stop()
        if self._progress_bar is None:
            return
        self._stop_event = threading.Event()
        self._progress_bar.set_postfix_str(f"{self._label} |", refresh=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=0.5)
        self._thread = None

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        frame_index = 0
        while not self._stop_event.wait(0.1):
            if self._progress_bar is None:
                return
            frame_index = (frame_index + 1) % len(frames)
            self._progress_bar.set_postfix_str(
                f"{self._label} {frames[frame_index]}",
                refresh=True,
            )


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_stem(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    safe = safe.strip("._-")
    if safe == "":
        return "model"
    return safe


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _discover_onnx_models(root_dir: str) -> List[str]:
    root_dir_abs = os.path.abspath(root_dir)
    pattern = os.path.join(root_dir_abs, "**", "*.onnx")
    return sorted(os.path.abspath(path) for path in glob.glob(pattern, recursive=True))


def _models_sha256(model_paths: List[str]) -> str:
    return _sha256_text("\n".join(str(path) for path in model_paths))


def _accuracy_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_accuracy_report.json",
    )


def _pytorch_accuracy_report_path(*, artifact_dir: str, model_path: str) -> str:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        artifact_dir,
        f"{model_stem}_pytorch_accuracy_report.json",
    )


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _classify_reports(
    *,
    tflite_report: Optional[Dict[str, Any]],
    pytorch_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    tflite_exists = tflite_report is not None
    pytorch_exists = pytorch_report is not None
    tflite_pass = (
        bool(tflite_report.get("evaluation_pass", False))
        if tflite_exists
        else None
    )
    pytorch_pass = (
        bool(pytorch_report.get("evaluation_pass", False))
        if pytorch_exists
        else None
    )

    if not tflite_exists and not pytorch_exists:
        return {
            "classification": "missing_both_reports",
            "strict_pass": False,
            "reason": "missing_tflite_report,missing_pytorch_report",
            "tflite_accuracy_pass": None,
            "pytorch_accuracy_pass": None,
        }
    if not tflite_exists:
        return {
            "classification": "missing_tflite_report",
            "strict_pass": False,
            "reason": "missing_tflite_report",
            "tflite_accuracy_pass": None,
            "pytorch_accuracy_pass": pytorch_pass,
        }
    if not pytorch_exists:
        return {
            "classification": "missing_pytorch_report",
            "strict_pass": False,
            "reason": "missing_pytorch_report",
            "tflite_accuracy_pass": tflite_pass,
            "pytorch_accuracy_pass": None,
        }
    if tflite_pass and pytorch_pass:
        return {
            "classification": "pass",
            "strict_pass": True,
            "reason": "",
            "tflite_accuracy_pass": True,
            "pytorch_accuracy_pass": True,
        }
    if not tflite_pass and not pytorch_pass:
        return {
            "classification": "both_fail",
            "strict_pass": False,
            "reason": "tflite_fail,pytorch_fail",
            "tflite_accuracy_pass": False,
            "pytorch_accuracy_pass": False,
        }
    if not tflite_pass:
        return {
            "classification": "tflite_fail",
            "strict_pass": False,
            "reason": "tflite_fail",
            "tflite_accuracy_pass": False,
            "pytorch_accuracy_pass": True,
        }
    return {
        "classification": "pytorch_fail",
        "strict_pass": False,
        "reason": "pytorch_fail",
        "tflite_accuracy_pass": True,
        "pytorch_accuracy_pass": False,
    }


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    entries = state.get("entries", [])
    counts = {
        "pass": 0,
        "skipped_model": 0,
        "missing_model": 0,
        "conversion_error": 0,
        "timeout": 0,
        "tflite_fail": 0,
        "pytorch_fail": 0,
        "both_fail": 0,
        "missing_tflite_report": 0,
        "missing_pytorch_report": 0,
        "missing_both_reports": 0,
    }
    for entry in entries:
        classification = str(entry.get("classification", "conversion_error"))
        counts[classification if classification in counts else "conversion_error"] += 1
    failed_entries = [
        entry
        for entry in entries
        if not bool(entry.get("strict_pass", False))
    ]
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "root_dir": state.get("root_dir", ""),
        "models_sha256": state.get("models_sha256", ""),
        "total_entries": int(len(entries)),
        "counts": counts,
        "strict_fail_count": int(len(failed_entries)),
        "failed_models": [
            {
                "model": str(entry.get("model", "")),
                "model_path": str(entry.get("model_path", "")),
                "classification": str(entry.get("classification", "")),
                "reason": str(entry.get("reason", "")),
            }
            for entry in failed_entries
        ],
        "generated_at": _utc_now_iso(),
    }


def _write_markdown_summary(path: str, *, state: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Flatbuffer Direct Bulk Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Root dir: `{summary['root_dir']}`")
    lines.append(f"- Total entries: {summary['total_entries']}")
    lines.append(f"- Strict fail count: {summary['strict_fail_count']}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Classification | Count |")
    lines.append("| --- | ---: |")
    for key, value in summary["counts"].items():
        lines.append(f"| `{key}` | {int(value)} |")
    lines.append("")
    lines.append("## Failed Models")
    lines.append("")
    if not summary["failed_models"]:
        lines.append("None")
    else:
        lines.append("| Model | Classification | Reason |")
        lines.append("| --- | --- | --- |")
        for failed in summary["failed_models"]:
            reason = str(failed.get("reason", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| `{failed.get('model_path', '')}` | "
                f"`{failed.get('classification', '')}` | {reason} |"
            )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| # | Model | Classification | Strict Pass | Exit | Reason |")
    lines.append("| ---: | --- | --- | :---: | ---: | --- |")
    for entry in state.get("entries", []):
        reason = str(entry.get("reason", "")).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {int(entry.get('index', 0))} | `{entry.get('model_path', '')}` | "
            f"`{entry.get('classification', '')}` | "
            f"{'Y' if bool(entry.get('strict_pass', False)) else 'N'} | "
            f"{entry.get('onnx2tf_exit_code', '')} | {reason} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_flatbuffer_direct_bulk_verification(
    *,
    root_dir: str,
    output_dir: str,
    resume: bool = False,
    onnx2tf_command: str = "",
    timeout_sec: int = 600,
    native_pytorch_generation_timeout_sec: int = 0,
    skip_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    root_dir_abs = os.path.abspath(root_dir)
    output_dir_abs = os.path.abspath(output_dir)
    runs_dir = os.path.join(output_dir_abs, _RUNS_DIRNAME)
    state_path = os.path.join(output_dir_abs, _STATE_FILENAME)
    summary_json_path = os.path.join(output_dir_abs, _SUMMARY_JSON_FILENAME)
    summary_md_path = os.path.join(output_dir_abs, _SUMMARY_MD_FILENAME)

    if not os.path.isdir(root_dir_abs):
        raise FileNotFoundError(f"Root directory does not exist. path={root_dir_abs}")

    models = _discover_onnx_models(root_dir_abs)
    models_sha256 = _models_sha256(models)
    normalized_skip_model_names = sorted(
        {
            os.path.basename(str(model_name)).strip()
            for model_name in (skip_model_names or [])
            if str(model_name).strip() != ""
        }
    )
    os.makedirs(runs_dir, exist_ok=True)

    if str(onnx2tf_command).strip() == "":
        local_onnx2tf_py = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "onnx2tf.py",
            )
        )
        command_prefix = [str(sys.executable), str(local_onnx2tf_py)]
    else:
        command_prefix = shlex.split(str(onnx2tf_command))
    if len(command_prefix) == 0:
        raise ValueError("onnx2tf command prefix must not be empty.")

    if resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if str(state.get("models_sha256", "")) != str(models_sha256):
            raise RuntimeError(
                "Resume state does not match the current discovered models. "
                f"state_sha256={state.get('models_sha256', '')} current_sha256={models_sha256}"
            )
        if list(state.get("skip_model_names", [])) != normalized_skip_model_names:
            raise RuntimeError(
                "Resume state does not match the current skip_model_names. "
                f"state_skip_model_names={state.get('skip_model_names', [])} "
                f"current_skip_model_names={normalized_skip_model_names}"
            )
        if int(state.get("native_pytorch_generation_timeout_sec", 0)) != int(native_pytorch_generation_timeout_sec):
            raise RuntimeError(
                "Resume state does not match the current native_pytorch_generation_timeout_sec. "
                f"state_timeout={state.get('native_pytorch_generation_timeout_sec', 0)} "
                f"current_timeout={int(native_pytorch_generation_timeout_sec)}"
            )
        entries: List[Dict[str, Any]] = list(state.get("entries", []))
    else:
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "root_dir": root_dir_abs,
            "models_sha256": models_sha256,
            "skip_model_names": normalized_skip_model_names,
            "native_pytorch_generation_timeout_sec": int(native_pytorch_generation_timeout_sec),
            "started_at": _utc_now_iso(),
            "entries": [],
        }
        entries = []

    start_index = int(len(entries))
    progress_bar = _create_progress_bar(
        total=len(models),
        initial=start_index,
        desc="flatbuffer_direct bulk",
    )
    spinner = _ProgressSpinner(progress_bar)
    try:
        for offset, model_path in enumerate(models[start_index:], start=start_index + 1):
            model_name = os.path.basename(model_path)
            run_dir = os.path.join(
                runs_dir,
                f"{int(offset):04d}_{_sanitize_stem(os.path.splitext(model_name)[0])}",
            )
            artifact_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            stdout_log_path = os.path.join(run_dir, "command.stdout.log")
            stderr_log_path = os.path.join(run_dir, "command.stderr.log")
            tflite_accuracy_report_path = _accuracy_report_path(
                artifact_dir=artifact_dir,
                model_path=model_name,
            )
            pytorch_accuracy_report_path = _pytorch_accuracy_report_path(
                artifact_dir=artifact_dir,
                model_path=model_name,
            )

            entry: Dict[str, Any] = {
                "index": int(offset),
                "model": str(model_name),
                "model_path": str(model_path),
                "run_dir": str(run_dir),
                "artifact_dir": str(artifact_dir),
                "stdout_log_path": str(stdout_log_path),
                "stderr_log_path": str(stderr_log_path),
                "tflite_accuracy_report_path": str(tflite_accuracy_report_path),
                "pytorch_accuracy_report_path": str(pytorch_accuracy_report_path),
                "started_at": _utc_now_iso(),
                "onnx2tf_exit_code": None,
                "classification": "",
                "strict_pass": False,
                "reason": "",
                "duration_sec": 0.0,
                "command": "",
                "tflite_accuracy_pass": None,
                "pytorch_accuracy_pass": None,
            }

            started = time.time()
            if model_name in normalized_skip_model_names:
                entry["classification"] = "skipped_model"
                entry["strict_pass"] = True
                entry["reason"] = "skipped_by_request"
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            if not os.path.exists(model_path):
                entry["classification"] = "missing_model"
                entry["strict_pass"] = False
                entry["reason"] = "model_not_found"
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            cmd = [
                *command_prefix,
                "-i",
                str(model_path),
                "-o",
                str(artifact_dir),
                "-tb",
                "flatbuffer_direct",
                "-cotof",
                "-fdopt",
                "-fdots",
                "-fdodo",
                "-fdoep",
            ]
            if int(native_pytorch_generation_timeout_sec) > 0:
                cmd.extend(
                    [
                        "--native_pytorch_generation_timeout_sec",
                        str(int(native_pytorch_generation_timeout_sec)),
                    ]
                )
            entry["command"] = shlex.join(cmd)

            spinner.set_context(
                model_name=model_name,
                status="running",
            )
            spinner.start()
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=run_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=int(timeout_sec),
                )
                stdout_text = completed.stdout if completed.stdout is not None else ""
                stderr_text = completed.stderr if completed.stderr is not None else ""
                entry["onnx2tf_exit_code"] = int(completed.returncode)
            except subprocess.TimeoutExpired as ex:
                spinner.stop()
                stdout_text = ex.stdout if isinstance(ex.stdout, str) else ""
                stderr_text = ex.stderr if isinstance(ex.stderr, str) else ""
                entry["classification"] = "timeout"
                entry["strict_pass"] = False
                entry["reason"] = f"timeout_after_{int(timeout_sec)}s"
                with open(stdout_log_path, "w", encoding="utf-8") as f:
                    f.write(stdout_text)
                with open(stderr_log_path, "w", encoding="utf-8") as f:
                    f.write(stderr_text)
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue
            finally:
                spinner.stop()

            with open(stdout_log_path, "w", encoding="utf-8") as f:
                f.write(stdout_text)
            with open(stderr_log_path, "w", encoding="utf-8") as f:
                f.write(stderr_text)

            if int(entry["onnx2tf_exit_code"]) != 0:
                entry["classification"] = "conversion_error"
                entry["strict_pass"] = False
                entry["reason"] = "onnx2tf_nonzero_exit"
                entry["duration_sec"] = float(time.time() - started)
                entries.append(entry)
                state["entries"] = entries
                _write_json(state_path, state)
                _update_progress_bar(
                    progress_bar,
                    model_name=model_name,
                    classification=str(entry["classification"]),
                )
                continue

            tflite_report = _read_json(tflite_accuracy_report_path)
            pytorch_report = _read_json(pytorch_accuracy_report_path)
            classification = _classify_reports(
                tflite_report=tflite_report,
                pytorch_report=pytorch_report,
            )
            entry["classification"] = str(classification["classification"])
            entry["strict_pass"] = bool(classification["strict_pass"])
            entry["reason"] = str(classification["reason"])
            entry["tflite_accuracy_pass"] = classification["tflite_accuracy_pass"]
            entry["pytorch_accuracy_pass"] = classification["pytorch_accuracy_pass"]
            entry["duration_sec"] = float(time.time() - started)
            entries.append(entry)
            state["entries"] = entries
            _write_json(state_path, state)
            _update_progress_bar(
                progress_bar,
                model_name=model_name,
                classification=str(entry["classification"]),
            )
    finally:
        spinner.stop()
        if progress_bar is not None:
            progress_bar.close()

    state["finished_at"] = _utc_now_iso()
    summary = _build_summary(state)
    state["summary"] = summary
    _write_json(state_path, state)
    _write_json(summary_json_path, summary)
    _write_markdown_summary(summary_md_path, state=state, summary=summary)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run bulk flatbuffer_direct verification using recursive ONNX discovery "
            "and onnx2tf -tb flatbuffer_direct -cotof -fdopt -fdots -fdodo -fdoep."
        )
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="flatbuffer_direct_bulk_report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--onnx2tf_command", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--native_pytorch_generation_timeout_sec", type=int, default=0)
    parser.add_argument(
        "--skip_model_name",
        "--skip_model_names",
        dest="skip_model_name",
        action="append",
        default=[],
        help="Basename of an ONNX model to skip during bulk verification. Repeatable.",
    )
    args = parser.parse_args()

    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(args.root_dir),
        output_dir=str(args.output_dir),
        resume=bool(args.resume),
        onnx2tf_command=str(args.onnx2tf_command),
        timeout_sec=int(args.timeout_sec),
        native_pytorch_generation_timeout_sec=int(args.native_pytorch_generation_timeout_sec),
        skip_model_names=list(args.skip_model_name),
    )
    summary = state.get("summary", {}) or {}
    failed_models = list(summary.get("failed_models", []))
    print(
        "Flatbuffer-direct bulk verification complete. "
        f"total_entries={len(state.get('entries', []))} "
        f"pass_count={summary.get('counts', {}).get('pass', 0)} "
        f"fail_count={summary.get('strict_fail_count', 0)}"
    )
    if failed_models:
        print("Failed ONNX models:")
        for failed in failed_models:
            print(
                f"- {failed.get('model_path', '')} "
                f"[{failed.get('classification', '')}] "
                f"{failed.get('reason', '')}"
            )
    raise SystemExit(1 if int(summary.get("strict_fail_count", 0)) > 0 else 0)


if __name__ == "__main__":
    main()
