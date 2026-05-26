"""Tests for core/agent.py — ReAct loop with mocked Anthropic API."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from complexity_oracle.core.agent import run_agent, _extract_text, _parse_structured_response
from complexity_oracle.models.analysis import AgentResult

# ── Mock factories ────────────────────────────────────────────────────────────

def _usage(input_tokens: int = 100, output_tokens: int = 80) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, inputs: dict, block_id: str = "tu_001") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=inputs, id=block_id)


def _response(stop_reason: str, content: list, usage: SimpleNamespace | None = None) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    r.usage = usage or _usage()
    return r


# Structured response text the mock agent returns
_MOCK_STRUCTURED_RESPONSE = (
    "VERDICT: This function is O(n²) in practice despite having one visible loop.\n"
    "WHY: The `item not in seen` check on line 5 is an O(n) list scan executed on "
    "every iteration of the outer loop, making true complexity O(n²).\n"
    "FIX: Replace `seen = []` with `seen = set()` for O(1) membership tests.\n"
    "CODE:\n"
    "seen = set()\n"
    "for item in data:\n"
    "    if item not in seen:\n"
    "        seen.add(item)"
)


# A two-turn mock conversation:
#  Turn 1 → tool_use (three tool calls)
#  Turn 2 → end_turn (structured text response)
def _two_turn_responses() -> list:
    turn1 = _response(
        stop_reason="tool_use",
        content=[
            _tool_use_block("analyze_ast",  {"file_path": "f.py"}, "tu_001"),
            _tool_use_block("run_profiler", {"file_path": "f.py", "function_name": "fn"}, "tu_002"),
            _tool_use_block("fit_curve",    {
                "input_sizes": [10, 100, 1000, 10000],
                "runtimes_ms": [0.001, 0.01, 0.1, 1.0],
                "timed_out": False,
                "static_complexity": "O(n)",
            }, "tu_003"),
        ],
        usage=_usage(200, 50),
    )
    turn2 = _response(
        stop_reason="end_turn",
        content=[_text_block(_MOCK_STRUCTURED_RESPONSE)],
        usage=_usage(300, 120),
    )
    return [turn1, turn2]


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestExtractText:
    def test_returns_text_block(self):
        blocks = [_text_block("Hello world")]
        assert _extract_text(blocks) == "Hello world"

    def test_empty_list(self):
        assert _extract_text([]) == ""

    def test_skips_non_text_blocks(self):
        blocks = [_tool_use_block("analyze_ast", {}), _text_block("Found it")]
        assert _extract_text(blocks) == "Found it"


class TestParseStructuredResponse:
    def test_verdict_extracted(self):
        result = _parse_structured_response(_MOCK_STRUCTURED_RESPONSE)
        assert result["verdict"] == "This function is O(n²) in practice despite having one visible loop."

    def test_why_extracted(self):
        result = _parse_structured_response(_MOCK_STRUCTURED_RESPONSE)
        assert "O(n) list scan" in result["why"]

    def test_fix_extracted(self):
        result = _parse_structured_response(_MOCK_STRUCTURED_RESPONSE)
        assert "set()" in result["fix"]

    def test_code_snippet_extracted(self):
        result = _parse_structured_response(_MOCK_STRUCTURED_RESPONSE)
        assert result["code_snippet"] is not None
        assert "seen = set()" in result["code_snippet"]

    def test_code_snippet_none_when_absent(self):
        text = "VERDICT: O(n).\nWHY: One loop.\nFIX: No change needed."
        result = _parse_structured_response(text)
        assert result["code_snippet"] is None

    def test_fallback_when_unstructured(self):
        text = "This function is O(n). Nothing to worry about."
        result = _parse_structured_response(text)
        assert result["verdict"] == "This function is O(n)"
        assert result["why"] == text  # full text surfaced in why


# ── Missing API key ───────────────────────────────────────────────────────────

class TestMissingApiKey:
    def test_raises_environment_error_when_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="No Anthropic API key"):
                run_agent("f.py", "fn")

    def test_error_message_mentions_setup(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="oracle setup"):
                run_agent("f.py", "fn")


# ── ReAct loop — mocked API ───────────────────────────────────────────────────

class TestReActLoop:
    @pytest.fixture(autouse=True)
    def set_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            yield

    def _mock_client(self, responses: list):
        """Patch Anthropic client so messages.create returns responses in order."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        return patch("complexity_oracle.core.agent.anthropic.Anthropic", return_value=mock_client)

    def test_returns_agent_result(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result, AgentResult)

    def test_verdict_is_non_empty_string(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0

    def test_why_is_non_empty_string(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.why, str)
        assert len(result.why) > 0

    def test_fix_is_non_empty_string(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.fix, str)
        assert len(result.fix) > 0

    def test_code_snippet_populated_from_mock(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert result.code_snippet is not None
        assert "seen = set()" in result.code_snippet

    def test_explanation_holds_raw_text(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.explanation, str)
        assert "VERDICT:" in result.explanation

    def test_tokens_used_is_positive_int(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.tokens_used, int)
        assert result.tokens_used > 0

    def test_tokens_accumulate_across_turns(self):
        # Turn 1: 200+50=250, Turn 2: 300+120=420 → total 670
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert result.tokens_used == 670

    def test_verdict_parsed_from_structured_response(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert "O(n²)" in result.verdict

    def test_tool_dispatch_called_for_each_tool(self):
        with self._mock_client(_two_turn_responses()):
            with patch("complexity_oracle.core.agent.dispatch_tool") as mock_dispatch:
                mock_dispatch.return_value = '{"ok": true}'
                run_agent("f.py", "fn")
        # Three tool calls in turn 1
        assert mock_dispatch.call_count == 3

    def test_tool_names_dispatched_correctly(self):
        with self._mock_client(_two_turn_responses()):
            with patch("complexity_oracle.core.agent.dispatch_tool") as mock_dispatch:
                mock_dispatch.return_value = '{"ok": true}'
                run_agent("f.py", "fn")
        dispatched = {call.args[0] for call in mock_dispatch.call_args_list}
        assert dispatched == {"analyze_ast", "run_profiler", "fit_curve"}

    def test_loop_terminates_on_end_turn(self):
        # Immediately returns end_turn — no tools called
        structured = (
            "VERDICT: This function is O(1). No loops present.\n"
            "WHY: The function performs a fixed number of operations regardless of input size.\n"
            "FIX: No change needed."
        )
        single_turn = [_response(stop_reason="end_turn", content=[_text_block(structured)])]
        with self._mock_client(single_turn):
            result = run_agent("f.py", "fn")
        assert "O(1)" in result.verdict

    def test_api_called_with_correct_model(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _two_turn_responses()
        with patch("complexity_oracle.core.agent.anthropic.Anthropic", return_value=mock_client):
            run_agent("f.py", "fn")
        call_kwargs = mock_client.messages.create.call_args_list[0].kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_api_called_with_tool_definitions(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _two_turn_responses()
        with patch("complexity_oracle.core.agent.anthropic.Anthropic", return_value=mock_client):
            run_agent("f.py", "fn")
        call_kwargs = mock_client.messages.create.call_args_list[0].kwargs
        tool_names = {t["name"] for t in call_kwargs["tools"]}
        assert tool_names == {"analyze_ast", "run_profiler", "fit_curve"}
