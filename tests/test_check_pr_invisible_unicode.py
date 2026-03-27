from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_pr_invisible_unicode.py"
MODULE_SPEC = importlib.util.spec_from_file_location("check_pr_invisible_unicode", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
check_pr_invisible_unicode = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = check_pr_invisible_unicode
MODULE_SPEC.loader.exec_module(check_pr_invisible_unicode)


def test_scan_unified_diff_ignores_safe_additions() -> None:
    diff_text = """\
diff --git a/sample.py b/sample.py
+++ b/sample.py
@@ -0,0 +1,2 @@
+print("safe")
+print("still safe")
"""
    findings = check_pr_invisible_unicode.scan_unified_diff(diff_text)
    assert findings == []


def test_scan_unified_diff_reports_dangerous_added_lines() -> None:
    diff_text = """\
diff --git a/sample.py b/sample.py
+++ b/sample.py
@@ -1 +1,3 @@
-print("old")
+print("start")
+print("bad\\u202evalue")
+print("also bad\\u200bvalue")
""".encode("utf-8").decode("unicode_escape")
    findings = check_pr_invisible_unicode.scan_unified_diff(diff_text)
    assert [(finding.path, finding.line_number, finding.codepoints) for finding in findings] == [
        ("sample.py", 2, (0x202E,)),
        ("sample.py", 3, (0x200B,)),
    ]


def test_scan_unified_diff_respects_allowlists() -> None:
    diff_text = """\
diff --git a/docs/guide.md b/docs/guide.md
+++ b/docs/guide.md
@@ -0,0 +1 @@
+hidden\\u2060text
diff --git a/src/config.yaml b/src/config.yaml
+++ b/src/config.yaml
@@ -0,0 +1 @@
+name: value\\ufeff
""".encode("utf-8").decode("unicode_escape")
    findings = check_pr_invisible_unicode.scan_unified_diff(
        diff_text,
        allowlist_paths=("docs/*",),
        allowlist_extensions=(".yaml",),
    )
    assert findings == []


def test_scan_unified_diff_ignores_binary_sections_and_removed_lines() -> None:
    diff_text = """\
diff --git a/image.png b/image.png
Binary files a/image.png and b/image.png differ
diff --git a/sample.py b/sample.py
+++ b/sample.py
@@ -4 +4 @@
-danger\\u202eold
+safe
""".encode("utf-8").decode("unicode_escape")
    findings = check_pr_invisible_unicode.scan_unified_diff(diff_text)
    assert findings == []
