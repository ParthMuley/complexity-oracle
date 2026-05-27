"""Tests for mcp/server.py — MCP tool registrations and SSE mount."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from complexity_oracle.api.app import app
from complexity_oracle.mcp.server import mcp, analyze_ast, run_profiler, fit_curve

api_client = TestClient(app, raise_server_exceptions=True)
# MCP SSE security validates the Host header — use localhost to pass it
mcp_client = TestClient(app, base_url="http://localhost", raise_server_exceptions=True)


# ── SSE mount (/mcp) ─────────────────────────────────────────────────────────

class TestSSEMount:
    """Verify the /mcp SSE sub-app is mounted on the FastAPI app.

    SSE connections are persistent streams that never close — making a live
    HTTP call in a test would hang. Instead we inspect the app's route table,
    which is sufficient to prove the mount is wired up correctly.
    """

    def test_mcp_path_is_mounted(self):
        mounted_paths = [str(r.path) for r in app.routes if hasattr(r, "path")]
        assert "/mcp" in mounted_paths

    def test_mcp_mount_is_an_asgi_app(self):
        from starlette.routing import Mount
        mcp_mount = next(
            (r for r in app.routes if hasattr(r, "path") and r.path == "/mcp"),
            None,
        )
        assert mcp_mount is not None
        assert isinstance(mcp_mount, Mount)


# ── Server identity ───────────────────────────────────────────────────────────

class TestServerIdentity:
    def test_server_name(self):
        assert mcp.name == "complexity-oracle"

    def test_server_has_three_tools(self):
        tools = mcp._tool_manager._tools
        assert len(tools) == 3

    def test_tool_names_registered(self):
        tools = mcp._tool_manager._tools
        assert set(tools.keys()) == {"analyze_ast", "run_profiler", "fit_curve"}


# ── analyze_ast tool ──────────────────────────────────────────────────────────

class TestAnalyzeAstTool:
    def test_returns_dict(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data):\n    for x in data:\n        pass\n")
        result = analyze_ast(file_path=str(f))
        assert isinstance(result, dict)

    def test_contains_static_complexity(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data):\n    for x in data:\n        pass\n")
        result = analyze_ast(file_path=str(f))
        assert "static_complexity" in result

    def test_contains_functions_list(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data): pass\n")
        result = analyze_ast(file_path=str(f))
        assert "functions" in result
        assert "foo" in result["functions"]

    def test_error_on_missing_file(self):
        result = analyze_ast(file_path="/nonexistent/path/file.py")
        assert "error" in result

    def test_delegates_to_adapter(self):
        with patch("complexity_oracle.mcp.server._adapter_analyze_ast") as mock:
            mock.return_value = {"static_complexity": "O(n)"}
            result = analyze_ast(file_path="f.py")
        mock.assert_called_once_with({"file_path": "f.py"})
        assert result == {"static_complexity": "O(n)"}


# ── run_profiler tool ─────────────────────────────────────────────────────────

class TestRunProfilerTool:
    def test_returns_dict(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data):\n    return sum(data)\n")
        result = run_profiler(file_path=str(f), function_name="foo")
        assert isinstance(result, dict)

    def test_contains_input_sizes(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data):\n    return sum(data)\n")
        result = run_profiler(file_path=str(f), function_name="foo")
        assert "input_sizes" in result

    def test_contains_runtimes(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(data):\n    return sum(data)\n")
        result = run_profiler(file_path=str(f), function_name="foo")
        assert "runtimes_ms" in result

    def test_default_timeout_forwarded(self):
        with patch("complexity_oracle.mcp.server._adapter_run_profiler") as mock:
            mock.return_value = {"input_sizes": [], "runtimes_ms": [], "timed_out": False, "error": None}
            run_profiler(file_path="f.py", function_name="fn")
        call_input = mock.call_args[0][0]
        assert call_input["timeout_s"] == 5.0

    def test_custom_timeout_forwarded(self):
        with patch("complexity_oracle.mcp.server._adapter_run_profiler") as mock:
            mock.return_value = {"input_sizes": [], "runtimes_ms": [], "timed_out": False, "error": None}
            run_profiler(file_path="f.py", function_name="fn", timeout_s=2.0)
        call_input = mock.call_args[0][0]
        assert call_input["timeout_s"] == 2.0

    def test_delegates_to_adapter(self):
        with patch("complexity_oracle.mcp.server._adapter_run_profiler") as mock:
            mock.return_value = {"ok": True}
            result = run_profiler(file_path="f.py", function_name="fn")
        mock.assert_called_once()
        assert result == {"ok": True}


# ── fit_curve tool ────────────────────────────────────────────────────────────

class TestFitCurveTool:
    _SIZES = [10, 100, 1000, 10000]
    _TIMES_ON = [0.01, 0.1, 1.0, 10.0]   # linear
    _TIMES_ON2 = [0.01, 1.0, 100.0, 10000.0]  # quadratic

    def test_returns_dict(self):
        result = fit_curve(
            input_sizes=self._SIZES,
            runtimes_ms=self._TIMES_ON,
            timed_out=False,
            static_complexity="O(n)",
        )
        assert isinstance(result, dict)

    def test_contains_empirical_complexity(self):
        result = fit_curve(
            input_sizes=self._SIZES,
            runtimes_ms=self._TIMES_ON,
            timed_out=False,
            static_complexity="O(n)",
        )
        assert "empirical_complexity" in result

    def test_contains_r_squared(self):
        result = fit_curve(
            input_sizes=self._SIZES,
            runtimes_ms=self._TIMES_ON,
            timed_out=False,
            static_complexity="O(n)",
        )
        assert "r_squared" in result

    def test_detects_on_for_linear_data(self):
        result = fit_curve(
            input_sizes=self._SIZES,
            runtimes_ms=self._TIMES_ON,
            timed_out=False,
            static_complexity="O(n)",
        )
        assert result["empirical_complexity"] == "O(n)"

    def test_detects_mismatch(self):
        result = fit_curve(
            input_sizes=self._SIZES,
            runtimes_ms=self._TIMES_ON2,
            timed_out=False,
            static_complexity="O(n)",
        )
        assert result["mismatch"] is True

    def test_delegates_to_adapter(self):
        with patch("complexity_oracle.mcp.server._adapter_fit_curve") as mock:
            mock.return_value = {"empirical_complexity": "O(n)", "r_squared": 0.99, "mismatch": False, "mismatch_reason": None}
            fit_curve(
                input_sizes=self._SIZES,
                runtimes_ms=self._TIMES_ON,
                timed_out=False,
                static_complexity="O(n)",
            )
        mock.assert_called_once_with({
            "input_sizes": self._SIZES,
            "runtimes_ms": self._TIMES_ON,
            "timed_out": False,
            "static_complexity": "O(n)",
        })
