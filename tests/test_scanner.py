"""Tests for core/scanner.py — folder scan and local import traversal."""
from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest

from complexity_oracle.core.scanner import (
    build_local_module_map,
    discover_python_files,
    find_local_imports,
    scan_folder,
)
from complexity_oracle.models.analysis import Complexity, FolderReport, FunctionResult


# ── discover_python_files ─────────────────────────────────────────────────────

class TestDiscoverPythonFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "foo.py").write_text("def foo(): pass")
        (tmp_path / "bar.py").write_text("def bar(): pass")
        files = discover_python_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert "foo.py" in basenames
        assert "bar.py" in basenames

    def test_ignores_non_py_files(self, tmp_path):
        (tmp_path / "foo.py").write_text("def foo(): pass")
        (tmp_path / "readme.txt").write_text("hello")
        files = discover_python_files(str(tmp_path))
        assert all(f.endswith(".py") for f in files)

    def test_skips_init_file(self, tmp_path):
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "foo.py").write_text("def foo(): pass")
        files = discover_python_files(str(tmp_path))
        assert not any("__init__" in f for f in files)

    def test_skips_main_file(self, tmp_path):
        (tmp_path / "__main__.py").write_text("")
        (tmp_path / "foo.py").write_text("def foo(): pass")
        files = discover_python_files(str(tmp_path))
        assert not any("__main__" in f for f in files)

    def test_skips_test_files(self, tmp_path):
        (tmp_path / "test_foo.py").write_text("def test_it(): pass")
        (tmp_path / "foo_test.py").write_text("def test_it(): pass")
        (tmp_path / "foo.py").write_text("def foo(): pass")
        files = discover_python_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert "foo.py" in basenames
        assert "test_foo.py" not in basenames
        assert "foo_test.py" not in basenames

    def test_non_recursive_ignores_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("def deep(): pass")
        (tmp_path / "top.py").write_text("def top(): pass")
        files = discover_python_files(str(tmp_path), recursive=False)
        basenames = {os.path.basename(f) for f in files}
        assert "top.py" in basenames
        assert "deep.py" not in basenames

    def test_recursive_finds_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("def deep(): pass")
        (tmp_path / "top.py").write_text("def top(): pass")
        files = discover_python_files(str(tmp_path), recursive=True)
        basenames = {os.path.basename(f) for f in files}
        assert "top.py" in basenames
        assert "deep.py" in basenames

    def test_returns_sorted_paths(self, tmp_path):
        (tmp_path / "z_last.py").write_text("def z(): pass")
        (tmp_path / "a_first.py").write_text("def a(): pass")
        files = discover_python_files(str(tmp_path))
        assert files == sorted(files)

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "foo.py").write_text("def foo(): pass")
        files = discover_python_files(str(tmp_path))
        assert all(os.path.isabs(f) for f in files)

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert discover_python_files(str(tmp_path)) == []


# ── build_local_module_map ────────────────────────────────────────────────────

class TestBuildLocalModuleMap:
    def test_maps_stem_to_path(self, tmp_path):
        f = tmp_path / "utils.py"
        f.write_text("")
        mapping = build_local_module_map([str(f)])
        assert "utils" in mapping
        assert mapping["utils"] == str(f)

    def test_multiple_files(self, tmp_path):
        files = []
        for name in ["utils.py", "helpers.py", "main.py"]:
            p = tmp_path / name
            p.write_text("")
            files.append(str(p))
        mapping = build_local_module_map(files)
        assert set(mapping.keys()) == {"utils", "helpers", "main"}

    def test_empty_list(self):
        assert build_local_module_map([]) == {}


# ── find_local_imports ────────────────────────────────────────────────────────

class TestFindLocalImports:
    def test_detects_import_statement(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("import utils\n")
        local = {"utils": str(tmp_path / "utils.py")}
        result = find_local_imports(str(f), local)
        assert "utils" in result

    def test_detects_from_import(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("from utils import helper\n")
        local = {"utils": str(tmp_path / "utils.py")}
        result = find_local_imports(str(f), local)
        assert "utils" in result

    def test_ignores_non_local_imports(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("import os\nimport json\nfrom pathlib import Path\n")
        local = {"utils": str(tmp_path / "utils.py")}
        result = find_local_imports(str(f), local)
        assert result == set()

    def test_returns_empty_on_syntax_error(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def (: pass")
        result = find_local_imports(str(f), {"utils": "utils.py"})
        assert result == set()

    def test_returns_empty_for_missing_file(self):
        result = find_local_imports("/nonexistent/file.py", {"utils": "utils.py"})
        assert result == set()


# ── scan_folder ───────────────────────────────────────────────────────────────

class TestScanFolder:
    def _write(self, tmp_path, name: str, code: str) -> str:
        f = tmp_path / name
        f.write_text(textwrap.dedent(code))
        return str(f)

    def test_returns_folder_report(self, tmp_path):
        self._write(tmp_path, "foo.py", "def foo(data):\n    return sum(data)\n")
        result = scan_folder(str(tmp_path))
        assert isinstance(result, FolderReport)

    def test_results_contain_function_results(self, tmp_path):
        self._write(tmp_path, "foo.py", "def foo(data):\n    return sum(data)\n")
        result = scan_folder(str(tmp_path))
        assert len(result.results) == 1
        assert isinstance(result.results[0], FunctionResult)

    def test_function_name_correct(self, tmp_path):
        self._write(tmp_path, "foo.py", "def my_func(data):\n    return sum(data)\n")
        result = scan_folder(str(tmp_path))
        assert result.results[0].function_name == "my_func"

    def test_scans_multiple_files(self, tmp_path):
        self._write(tmp_path, "a.py", "def fa(data):\n    return sum(data)\n")
        self._write(tmp_path, "b.py", "def fb(data):\n    return sum(data)\n")
        result = scan_folder(str(tmp_path))
        assert len(result.results) == 2

    def test_total_files_count(self, tmp_path):
        self._write(tmp_path, "a.py", "def fa(data): pass\n")
        self._write(tmp_path, "b.py", "def fb(data): pass\n")
        result = scan_folder(str(tmp_path))
        assert result.total_files == 2

    def test_skips_file_with_syntax_error(self, tmp_path):
        self._write(tmp_path, "bad.py", "def (: pass\n")
        self._write(tmp_path, "good.py", "def foo(data):\n    return sum(data)\n")
        result = scan_folder(str(tmp_path))
        assert len(result.skipped_files) == 1
        assert "bad.py" in result.skipped_files[0]

    def test_skips_file_with_no_functions(self, tmp_path):
        self._write(tmp_path, "constants.py", "X = 42\nY = 100\n")
        result = scan_folder(str(tmp_path))
        assert len(result.results) == 0
        assert len(result.skipped_files) == 0  # silence, not an error

    def test_folder_path_stored(self, tmp_path):
        self._write(tmp_path, "foo.py", "def foo(data): pass\n")
        result = scan_folder(str(tmp_path))
        assert os.path.isabs(result.folder_path)

    def test_mismatch_detected_for_on2_function(self, tmp_path):
        # hidden O(n²): `not in` on list
        self._write(tmp_path, "dup.py",
            "def remove_dups(data):\n"
            "    seen = []\n"
            "    for x in data:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return seen\n"
        )
        result = scan_folder(str(tmp_path))
        assert len(result.results) == 1
        assert result.results[0].mismatch is True

    def test_no_mismatch_for_linear_function(self, tmp_path):
        self._write(tmp_path, "lin.py",
            "def running_total(data):\n"
            "    total = 0\n"
            "    result = []\n"
            "    for x in data:\n"
            "        total += x\n"
            "        result.append(total)\n"
            "    return result\n"
        )
        result = scan_folder(str(tmp_path))
        assert len(result.results) == 1
        assert result.results[0].mismatch is False
