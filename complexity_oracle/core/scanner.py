"""Complexity Oracle — folder scanner with local import traversal.

Scans a directory for Python files, analyses every function in each file,
and produces a FolderReport.

Import traversal: before analysing each file the scanner builds a map of
all local module names (one per .py file in the folder).  Any 'unresolved
call' that resolves to a local module is re-annotated so the developer
knows it is handled by a sibling file, not an opaque external library.
"""
from __future__ import annotations

import ast
import os

from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report
from complexity_oracle.models.analysis import (
    Complexity,
    FolderReport,
    FunctionResult,
)


# ── File discovery ────────────────────────────────────────────────────────────

def discover_python_files(folder_path: str, recursive: bool = False) -> list[str]:
    """Return sorted absolute paths to all .py files under folder_path.

    Skips __init__.py, __main__.py, and test files (test_*.py / *_test.py)
    since those rarely contain functions meant to be profiled.

    Args:
        folder_path: Directory to search.
        recursive:   If True, descend into subdirectories.

    Returns:
        Sorted list of absolute .py file paths.
    """
    folder_path = os.path.abspath(folder_path)
    results: list[str] = []

    if recursive:
        for root, _dirs, files in os.walk(folder_path):
            for name in files:
                if _is_analysable(name):
                    results.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder_path):
            full = os.path.join(folder_path, name)
            if os.path.isfile(full) and _is_analysable(name):
                results.append(full)

    return sorted(results)


def _is_analysable(filename: str) -> bool:
    """Return True if this file should be included in a folder scan."""
    if not filename.endswith(".py"):
        return False
    stem = filename[:-3]
    if stem in ("__init__", "__main__"):
        return False
    if stem.startswith("test_") or stem.endswith("_test"):
        return False
    return True


# ── Local import resolution ───────────────────────────────────────────────────

def build_local_module_map(file_paths: list[str]) -> dict[str, str]:
    """Build a {module_stem: absolute_file_path} map from a list of files.

    Example: ["/proj/utils.py", "/proj/main.py"]
             → {"utils": "/proj/utils.py", "main": "/proj/main.py"}
    """
    return {
        os.path.splitext(os.path.basename(p))[0]: os.path.abspath(p)
        for p in file_paths
    }


def find_local_imports(file_path: str, local_modules: dict[str, str]) -> set[str]:
    """Return module stems that this file imports from the local module map.

    Reads the file's AST import statements and intersects with local_modules.
    Returns an empty set if the file cannot be parsed.

    Args:
        file_path:     Path to the Python file to inspect.
        local_modules: {module_stem: file_path} map built from the folder.

    Returns:
        Set of local module stems imported by this file.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (OSError, SyntaxError):
        return set()

    imported_local: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in local_modules:
                    imported_local.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in local_modules:
                    imported_local.add(root)
    return imported_local


# ── Folder scan ───────────────────────────────────────────────────────────────

def scan_folder(
    folder_path: str,
    recursive: bool = False,
    timeout_s: float = 5.0,
) -> FolderReport:
    """Scan all analysable Python files in folder_path.

    For each file, every function is parsed, profiled, and curve-fitted.
    Unresolved calls that point to sibling files in the folder are
    re-annotated as local (not external).

    Args:
        folder_path: Directory to scan.
        recursive:   Descend into subdirectories when True.
        timeout_s:   Per-input-size profiling timeout in seconds.

    Returns:
        FolderReport aggregating all per-function results.
    """
    py_files = discover_python_files(folder_path, recursive=recursive)
    local_modules = build_local_module_map(py_files)

    results: list[FunctionResult] = []
    skipped: list[str] = []
    seen_files: set[str] = set()

    for file_path in py_files:
        seen_files.add(file_path)
        imported_local = find_local_imports(file_path, local_modules)

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            parse = parse_file(file_path)
        except SyntaxError as e:
            skipped.append(f"{os.path.basename(file_path)}: {e}")
            continue
        except OSError as e:
            skipped.append(f"{os.path.basename(file_path)}: {e}")
            continue

        if not parse.functions:
            continue  # no functions to analyse — skip silently

        # ── Analyse each function ─────────────────────────────────────────────
        for fn_name in parse.functions:
            results.append(
                _analyse_function(
                    file_path=file_path,
                    function_name=fn_name,
                    parse=parse,
                    imported_local=imported_local,
                    timeout_s=timeout_s,
                )
            )

    return FolderReport(
        folder_path=os.path.abspath(folder_path),
        results=results,
        total_files=len(seen_files),
        skipped_files=skipped,
    )


def _analyse_function(
    file_path: str,
    function_name: str,
    parse,
    imported_local: set[str],
    timeout_s: float,
) -> FunctionResult:
    """Run profile → fit → report for one function and return a FunctionResult."""
    try:
        profile = profile_file(file_path, function_name, timeout_s=timeout_s)
        fit = fit_curve(profile, parse)
        report = build_report(file_path, parse, profile, fit)
    except Exception as e:  # noqa: BLE001
        return FunctionResult(
            file_path=file_path,
            function_name=function_name,
            static_complexity=parse.static_complexity,
            empirical_complexity=Complexity.UNKNOWN,
            r_squared=0.0,
            mismatch=False,
            warnings=[],
            error=str(e),
        )

    # Re-annotate warnings: replace "External import" with local-import note
    # for calls that come from sibling files in this folder scan.
    warnings = _annotate_local_warnings(report.warnings, parse, imported_local)

    return FunctionResult(
        file_path=file_path,
        function_name=function_name,
        static_complexity=report.parse.static_complexity,
        empirical_complexity=report.fit.empirical_complexity,
        r_squared=report.fit.r_squared,
        mismatch=report.fit.mismatch,
        warnings=warnings,
        error=None,
    )


def _annotate_local_warnings(
    warnings: list[str],
    parse,
    imported_local: set[str],
) -> list[str]:
    """Replace generic 'unresolved external call' warnings with local-import notes
    when the call's module is a sibling file in the current folder scan.
    """
    if not imported_local or not parse.unresolved_calls:
        return warnings

    local_call_names: set[str] = set()
    for call in parse.unresolved_calls:
        root = call.name.split(".")[0]
        if root in imported_local:
            local_call_names.add(call.name)

    if not local_call_names:
        return warnings

    updated: list[str] = []
    for w in warnings:
        # The report builder fires a single count-based warning for unresolved calls.
        # Leave other warnings untouched.
        if "unresolved external call" in w:
            external_count = sum(
                1 for c in parse.unresolved_calls
                if c.name.split(".")[0] not in imported_local
            )
            if external_count == 0:
                # All unresolved calls are actually local — suppress the warning.
                continue
            updated.append(
                f"{external_count} unresolved external call(s) — complexity may be underestimated"
            )
        else:
            updated.append(w)
    return updated
