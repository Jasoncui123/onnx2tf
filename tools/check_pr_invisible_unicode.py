#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath


DANGEROUS_CODEPOINTS = {
    0x00AD: "SOFT HYPHEN",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2060: "WORD JOINER",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0xFEFF: "ZERO WIDTH NO-BREAK SPACE",
}
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class Finding:
    path: str
    line_number: int
    codepoints: tuple[int, ...]


def parse_csv_items(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def is_allowlisted(
    path: str,
    path_patterns: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> bool:
    posix_path = PurePosixPath(path).as_posix()
    if any(posix_path.endswith(suffix) for suffix in suffixes):
        return True
    return any(fnmatch.fnmatch(posix_path, pattern) for pattern in path_patterns)


def scan_unified_diff(
    diff_text: str,
    *,
    allowlist_paths: tuple[str, ...] = (),
    allowlist_extensions: tuple[str, ...] = (),
) -> list[Finding]:
    findings: list[Finding] = []
    current_path: str | None = None
    current_line_number: int | None = None

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            if path == "/dev/null":
                current_path = None
                current_line_number = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            current_path = path
            current_line_number = None
            continue

        hunk_match = HUNK_RE.match(line)
        if hunk_match:
            current_line_number = int(hunk_match.group(1))
            continue

        if (
            current_path is None
            or current_line_number is None
            or is_allowlisted(current_path, allowlist_paths, allowlist_extensions)
        ):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            codepoints = tuple(
                dict.fromkeys(
                    ord(char)
                    for char in line[1:]
                    if ord(char) in DANGEROUS_CODEPOINTS
                )
            )
            if codepoints:
                findings.append(
                    Finding(
                        path=current_path,
                        line_number=current_line_number,
                        codepoints=codepoints,
                    )
                )
            current_line_number += 1
            continue

        if line.startswith(" "):
            current_line_number += 1

    return findings


def build_diff(base_ref: str, head_ref: str) -> str:
    command = [
        "git",
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "--find-renames",
        "--diff-filter=ACMRT",
        f"{base_ref}...{head_ref}",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr or "git diff failed")
    return result.stdout


def format_codepoint(codepoint: int) -> str:
    name = DANGEROUS_CODEPOINTS.get(codepoint) or unicodedata.name(chr(codepoint))
    return f"U+{codepoint:04X} {name}"


def emit_github_error(finding: Finding) -> None:
    codepoints = ", ".join(format_codepoint(codepoint) for codepoint in finding.codepoints)
    print(
        f"::error file={finding.path},line={finding.line_number}::"
        f"dangerous invisible unicode detected: {codepoints}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when a pull request adds dangerous invisible Unicode characters."
    )
    parser.add_argument("base_ref", help="Base git ref or commit SHA.")
    parser.add_argument("head_ref", help="Head git ref or commit SHA.")
    parser.add_argument(
        "--allowlist-paths",
        default=os.environ.get("INVISIBLE_UNICODE_ALLOWLIST_PATHS", ""),
        help="Comma-separated path globs to skip.",
    )
    parser.add_argument(
        "--allowlist-extensions",
        default=os.environ.get("INVISIBLE_UNICODE_ALLOWLIST_EXTENSIONS", ""),
        help="Comma-separated filename suffixes to skip.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    allowlist_paths = parse_csv_items(args.allowlist_paths)
    allowlist_extensions = parse_csv_items(args.allowlist_extensions)

    try:
        diff_text = build_diff(args.base_ref, args.head_ref)
    except RuntimeError as exc:
        print(f"failed to build diff: {exc}", file=sys.stderr)
        return 2

    findings = scan_unified_diff(
        diff_text,
        allowlist_paths=allowlist_paths,
        allowlist_extensions=allowlist_extensions,
    )

    if not findings:
        print("No dangerous invisible Unicode characters found in added lines.")
        return 0

    for finding in findings:
        emit_github_error(finding)

    print(
        f"Detected {len(findings)} added line(s) containing dangerous invisible Unicode characters.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
