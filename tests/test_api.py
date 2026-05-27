"""Tests for api/app.py — FastAPI service."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from complexity_oracle.api.app import app
from complexity_oracle.models.analysis import (
    AgentResult,
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
    Report,
    UnresolvedCall,
)

client = TestClient(app, raise_server_exceptions=True)

# ── Shared mock factories ─────────────────────────────────────────────────────

def _parse(
    functions: list[str] | None = None,
    complexity: Complexity = Complexity.O_N,
) -> ParseResult:
    return ParseResult(
        functions=functions if functions is not None else ["my_func"],
        max_loop_depth=1,
        static_complexity=complexity,
        unresolved_calls=[],
        flagged_lines={},
    )


def _profile() -> ProfileResult:
    return ProfileResult(
        input_sizes=[10, 100, 1000, 10000],
        runtimes_ms=[0.01, 0.1, 1.0, 10.0],
        timed_out=False,
        error=None,
    )


def _fit(
    complexity: Complexity = Complexity.O_N,
    mismatch: bool = False,
) -> FitResult:
    return FitResult(
        empirical_complexity=complexity,
        r_squared=0.999,
        mismatch=mismatch,
        mismatch_reason="static said O(n), empirical says O(n²)" if mismatch else None,
    )


def _report(parse: ParseResult | None = None, fit: FitResult | None = None) -> Report:
    return Report(
        source_file="/tmp/oracle_tmp.py",
        parse=parse or _parse(),
        profile=_profile(),
        fit=fit or _fit(),
        warnings=[],
    )


def _agent_result() -> AgentResult:
    return AgentResult(
        verdict="This function is O(n) in practice.",
        why="One loop over n items with O(1) operations inside.",
        fix="No change needed.",
        code_snippet=None,
        tokens_used=300,
        explanation="VERDICT: This function is O(n) in practice.",
    )


# Patch target for all pipeline functions inside the api module
_PIPELINE = "complexity_oracle.api.app"

# ── GET / — Web UI ────────────────────────────────────────────────────────────

class TestWebUI:
    def test_get_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_get_root_content_type_is_html(self):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_get_root_contains_form(self):
        r = client.get("/")
        assert "<form" in r.text

    def test_get_root_contains_code_textarea(self):
        r = client.get("/")
        assert "<textarea" in r.text

    def test_get_root_contains_api_key_input(self):
        r = client.get("/")
        assert 'type="password"' in r.text


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ── POST /analyze — happy path ────────────────────────────────────────────────

class TestAnalyzeHappyPath:
    """Full pipeline with agent disabled — mocks all core functions."""

    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()) as mp,
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            yield mp

    def test_returns_200(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.status_code == 200

    def test_response_has_function_name(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["function_name"] == "my_func"

    def test_response_has_static_complexity(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["static_complexity"] == "O(n)"

    def test_response_has_empirical_complexity(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["empirical_complexity"] == "O(n)"

    def test_response_has_r_squared(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["r_squared"] == pytest.approx(0.999)

    def test_response_has_mismatch_false(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["mismatch"] is False

    def test_response_has_warnings_list(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert isinstance(r.json()["warnings"], list)

    def test_agent_none_when_no_agent_true(self):
        r = client.post("/analyze", json={"code": "def my_func(x): pass", "no_agent": True})
        assert r.json()["agent"] is None

    def test_explicit_function_name_used(self):
        r = client.post("/analyze", json={
            "code": "def my_func(x): pass",
            "function_name": "my_func",
            "no_agent": True,
        })
        assert r.status_code == 200
        assert r.json()["function_name"] == "my_func"


# ── POST /analyze — mismatch response ─────────────────────────────────────────

class TestAnalyzeMismatch:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse(complexity=Complexity.O_N)),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit(complexity=Complexity.O_N2, mismatch=True)),
            patch(f"{_PIPELINE}.build_report", return_value=_report(
                parse=_parse(complexity=Complexity.O_N),
                fit=_fit(complexity=Complexity.O_N2, mismatch=True),
            )),
        ):
            yield

    def test_mismatch_true_in_response(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.json()["mismatch"] is True

    def test_mismatch_reason_present(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.json()["mismatch_reason"] is not None


# ── POST /analyze — agent enabled ─────────────────────────────────────────────

class TestAnalyzeWithAgent:
    @pytest.fixture(autouse=True)
    def mock_pipeline_and_agent(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
            patch(f"{_PIPELINE}.run_agent", return_value=_agent_result()),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        ):
            yield

    def test_returns_200(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.status_code == 200

    def test_agent_field_present(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.json()["agent"] is not None

    def test_agent_verdict_present(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.json()["agent"]["verdict"] == "This function is O(n) in practice."

    def test_agent_why_present(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert "One loop" in r.json()["agent"]["why"]

    def test_agent_tokens_used_present(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.json()["agent"]["tokens_used"] == 300

    def test_agent_code_snippet_null_when_none(self):
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.json()["agent"]["code_snippet"] is None


# ── POST /analyze — error cases ───────────────────────────────────────────────

class TestAnalyzeErrors:
    def test_syntax_error_returns_400(self):
        with patch(f"{_PIPELINE}.parse_file", side_effect=SyntaxError("invalid syntax")):
            r = client.post("/analyze", json={"code": "def f(: pass", "no_agent": True})
        assert r.status_code == 400
        assert "Syntax error" in r.json()["detail"]

    def test_no_functions_returns_400(self):
        with patch(f"{_PIPELINE}.parse_file", return_value=_parse(functions=[])):
            r = client.post("/analyze", json={"code": "x = 1", "no_agent": True})
        assert r.status_code == 400
        assert "No functions" in r.json()["detail"]

    def test_multiple_functions_no_name_returns_400(self):
        with patch(f"{_PIPELINE}.parse_file", return_value=_parse(functions=["foo", "bar"])):
            r = client.post("/analyze", json={"code": "def foo(): pass\ndef bar(): pass", "no_agent": True})
        assert r.status_code == 400
        assert "Multiple functions" in r.json()["detail"]

    def test_multiple_functions_with_name_ok(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse(functions=["foo", "bar"])),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            r = client.post("/analyze", json={
                "code": "def foo(): pass\ndef bar(): pass",
                "function_name": "foo",
                "no_agent": True,
            })
        assert r.status_code == 200

    def test_no_key_anywhere_with_agent_returns_401(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.status_code == 401
        assert "X-Anthropic-API-Key" in r.json()["detail"]


# ── Request defaults ───────────────────────────────────────────────────────────

class TestRequestDefaults:
    """Verify that omitting optional fields uses correct defaults."""

    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()) as mp,
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()) as mpr,
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            yield mp, mpr

    def test_minimal_body_accepted(self):
        # Only required field is `code`; no_agent defaults to False but no key →503
        # Test with no_agent=True to avoid the 503
        r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.status_code == 200

    def test_default_timeout_passed_to_profiler(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()) as mock_profiler,
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
            call_kwargs = mock_profiler.call_args.kwargs
            assert call_kwargs.get("timeout_s") == 5.0

    def test_custom_timeout_passed_to_profiler(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()) as mock_profiler,
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True, "timeout_s": 2.5})
            call_kwargs = mock_profiler.call_args.kwargs
            assert call_kwargs.get("timeout_s") == 2.5


# ── Key resolution ─────────────────────────────────────────────────────────────

class TestKeyResolution:
    """X-Anthropic-API-Key header takes precedence over env var; neither → 401."""

    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            yield

    def test_header_key_used_when_provided(self):
        with patch(f"{_PIPELINE}.run_agent", return_value=_agent_result()) as mock_agent:
            r = client.post(
                "/analyze",
                json={"code": "def f(x): pass", "no_agent": False},
                headers={"X-Anthropic-API-Key": "sk-ant-header"},
            )
        assert r.status_code == 200
        mock_agent.assert_called_once()
        _, call_kwargs = mock_agent.call_args
        assert call_kwargs.get("api_key") == "sk-ant-header"

    def test_env_key_used_as_fallback(self):
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env"}),
            patch(f"{_PIPELINE}.run_agent", return_value=_agent_result()) as mock_agent,
        ):
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.status_code == 200
        mock_agent.assert_called_once()
        _, call_kwargs = mock_agent.call_args
        assert call_kwargs.get("api_key") == "sk-ant-env"

    def test_header_key_takes_precedence_over_env(self):
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env"}),
            patch(f"{_PIPELINE}.run_agent", return_value=_agent_result()) as mock_agent,
        ):
            r = client.post(
                "/analyze",
                json={"code": "def f(x): pass", "no_agent": False},
                headers={"X-Anthropic-API-Key": "sk-ant-header"},
            )
        assert r.status_code == 200
        _, call_kwargs = mock_agent.call_args
        assert call_kwargs.get("api_key") == "sk-ant-header"

    def test_neither_key_returns_401(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.status_code == 401
        assert "X-Anthropic-API-Key" in r.json()["detail"]

    def test_no_key_needed_when_no_agent(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.status_code == 200


# ── CLOUD_MODE ─────────────────────────────────────────────────────────────────

class TestCloudMode:
    """CLOUD_MODE skips profiling, sets profiling_disabled=True in response."""

    @pytest.fixture(autouse=True)
    def mock_parse_report(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
        ):
            yield

    def test_cloud_mode_returns_200(self):
        with patch.dict(os.environ, {"CLOUD_MODE": "true"}):
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.status_code == 200

    def test_cloud_mode_profiling_disabled_true(self):
        with patch.dict(os.environ, {"CLOUD_MODE": "true"}):
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.json()["profiling_disabled"] is True

    def test_normal_mode_profiling_disabled_false(self):
        with (
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("CLOUD_MODE", None)
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        assert r.json()["profiling_disabled"] is False

    def test_cloud_mode_does_not_call_profile_file(self):
        with (
            patch(f"{_PIPELINE}.profile_file") as mock_profiler,
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch.dict(os.environ, {"CLOUD_MODE": "true"}),
        ):
            client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        mock_profiler.assert_not_called()

    def test_cloud_mode_does_not_call_fit_curve(self):
        with (
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve") as mock_fit,
            patch.dict(os.environ, {"CLOUD_MODE": "true"}),
        ):
            client.post("/analyze", json={"code": "def f(x): pass", "no_agent": True})
        mock_fit.assert_not_called()

    def test_agent_error_surfaced_when_agent_raises(self):
        with (
            patch(f"{_PIPELINE}.parse_file", return_value=_parse()),
            patch(f"{_PIPELINE}.profile_file", return_value=_profile()),
            patch(f"{_PIPELINE}.fit_curve", return_value=_fit()),
            patch(f"{_PIPELINE}.build_report", return_value=_report()),
            patch(f"{_PIPELINE}.run_agent", side_effect=RuntimeError("bad API key")),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-bad"}),
        ):
            r = client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        assert r.status_code == 200
        assert r.json()["agent"] is None
        assert "bad API key" in r.json()["agent_error"]

    def test_cloud_mode_agent_receives_cloud_mode_flag(self):
        with (
            patch(f"{_PIPELINE}.run_agent", return_value=_agent_result()) as mock_agent,
            patch.dict(os.environ, {"CLOUD_MODE": "true", "ANTHROPIC_API_KEY": "sk-ant-test"}),
        ):
            client.post("/analyze", json={"code": "def f(x): pass", "no_agent": False})
        _, call_kwargs = mock_agent.call_args
        assert call_kwargs.get("cloud_mode") is True
