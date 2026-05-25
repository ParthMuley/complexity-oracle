"""Tests for mcp/tools.py — tool schemas and JSON adapters."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from complexity_oracle.mcp.tools import (
    TOOL_DEFINITIONS,
    dispatch_tool,
    _adapter_analyze_ast,
    _adapter_run_profiler,
    _adapter_fit_curve,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_temp(source: str) -> str:
    """Write source to a temp file, return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    f.write(source)
    f.close()
    return f.name


LINEAR_SOURCE = """\
def scan(items):
    total = 0
    for item in items:
        total += item
    return total
"""

QUADRATIC_SOURCE = """\
def bubble(items):
    arr = list(items)
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""


# ── Tool definitions schema ───────────────────────────────────────────────────

class TestToolDefinitions:
    def test_has_three_tools(self):
        assert len(TOOL_DEFINITIONS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {"analyze_ast", "run_profiler", "fit_curve"}

    def test_each_tool_has_description(self):
        for tool in TOOL_DEFINITIONS:
            assert "description" in tool
            assert len(tool["description"]) > 10

    def test_each_tool_has_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_analyze_ast_requires_file_path(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "analyze_ast")
        assert "file_path" in schema["input_schema"]["required"]

    def test_run_profiler_required_fields(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "run_profiler")
        required = schema["input_schema"]["required"]
        assert "file_path" in required
        assert "function_name" in required

    def test_fit_curve_required_fields(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "fit_curve")
        required = schema["input_schema"]["required"]
        assert "input_sizes" in required
        assert "runtimes_ms" in required
        assert "static_complexity" in required


# ── analyze_ast adapter ───────────────────────────────────────────────────────

class TestAnalyzeAstAdapter:
    def test_returns_dict(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_analyze_ast({"file_path": path})
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_contains_static_complexity(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_analyze_ast({"file_path": path})
            assert "static_complexity" in result
            assert isinstance(result["static_complexity"], str)
        finally:
            os.unlink(path)

    def test_contains_functions(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_analyze_ast({"file_path": path})
            assert "functions" in result
            assert "scan" in result["functions"]
        finally:
            os.unlink(path)

    def test_contains_unresolved_calls(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_analyze_ast({"file_path": path})
            assert "unresolved_calls" in result
        finally:
            os.unlink(path)

    def test_missing_file_returns_error_dict(self):
        result = _adapter_analyze_ast({"file_path": "/nonexistent/file.py"})
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_missing_file_does_not_raise(self):
        # Should not raise — returns error dict
        result = _adapter_analyze_ast({"file_path": "/nonexistent/file.py"})
        assert isinstance(result, dict)


# ── run_profiler adapter ──────────────────────────────────────────────────────

class TestRunProfilerAdapter:
    def test_returns_dict(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_run_profiler({"file_path": path, "function_name": "scan"})
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_contains_runtimes_ms(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_run_profiler({"file_path": path, "function_name": "scan"})
            assert "runtimes_ms" in result
        finally:
            os.unlink(path)

    def test_contains_input_sizes(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = _adapter_run_profiler({"file_path": path, "function_name": "scan"})
            assert "input_sizes" in result
        finally:
            os.unlink(path)

    def test_bad_file_returns_error_in_dict(self):
        result = _adapter_run_profiler(
            {"file_path": "/nonexistent.py", "function_name": "f"}
        )
        assert "error" in result
        assert result["error"] is not None


# ── fit_curve adapter ─────────────────────────────────────────────────────────

class TestFitCurveAdapter:
    def test_returns_dict(self):
        result = _adapter_fit_curve({
            "input_sizes": [10, 100, 1000, 10000],
            "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
            "timed_out": False,
            "static_complexity": "O(n)",
        })
        assert isinstance(result, dict)

    def test_contains_empirical_complexity(self):
        result = _adapter_fit_curve({
            "input_sizes": [10, 100, 1000, 10000],
            "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
            "timed_out": False,
            "static_complexity": "O(n)",
        })
        assert "empirical_complexity" in result

    def test_contains_mismatch(self):
        result = _adapter_fit_curve({
            "input_sizes": [10, 100, 1000, 10000],
            "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
            "timed_out": False,
            "static_complexity": "O(n)",
        })
        assert "mismatch" in result

    def test_unknown_static_complexity_string_handled(self):
        result = _adapter_fit_curve({
            "input_sizes": [10, 100, 1000, 10000],
            "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
            "timed_out": False,
            "static_complexity": "some_garbage_value",
        })
        # Should fall back to UNKNOWN, not raise
        assert "empirical_complexity" in result


# ── dispatch_tool ─────────────────────────────────────────────────────────────

class TestDispatchTool:
    def test_returns_json_string(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = dispatch_tool("analyze_ast", {"file_path": path})
            assert isinstance(result, str)
            json.loads(result)  # must be valid JSON
        finally:
            os.unlink(path)

    def test_dispatches_analyze_ast(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            result = json.loads(dispatch_tool("analyze_ast", {"file_path": path}))
            assert "static_complexity" in result
        finally:
            os.unlink(path)

    def test_dispatches_fit_curve(self):
        result = json.loads(dispatch_tool("fit_curve", {
            "input_sizes": [10, 100, 1000, 10000],
            "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
            "timed_out": False,
            "static_complexity": "O(n)",
        }))
        assert "empirical_complexity" in result

    def test_unknown_tool_returns_error_json(self):
        result = json.loads(dispatch_tool("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_all_outputs_json_serialisable(self):
        path = _write_temp(LINEAR_SOURCE)
        try:
            for tool_name in ["analyze_ast"]:
                out = dispatch_tool(tool_name, {"file_path": path})
                json.loads(out)  # must not raise
        finally:
            os.unlink(path)
