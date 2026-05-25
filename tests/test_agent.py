"""Tests for core/agent.py — ReAct loop with mocked Anthropic API."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from complexity_oracle.core.agent import run_agent, _extract_text, _extract_verdict
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


# A two-turn mock conversation:
#  Turn 1 → tool_use (three tool calls)
#  Turn 2 → end_turn (text explanation)
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
        content=[_text_block(
            "This function is O(n²) in practice despite having one visible loop. "
            "The `item not in seen` check is an O(n) list scan on every iteration. "
            "Fix: replace the list with a set for O(1) membership checks."
        )],
        usage=_usage(300, 120),
    )
    return [turn1, turn2]


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_extract_text_returns_text_block(self):
        blocks = [_text_block("Hello world")]
        assert _extract_text(blocks) == "Hello world"

    def test_extract_text_empty_list(self):
        assert _extract_text([]) == ""

    def test_extract_text_skips_non_text_blocks(self):
        blocks = [_tool_use_block("analyze_ast", {}), _text_block("Found it")]
        assert _extract_text(blocks) == "Found it"

    def test_extract_verdict_first_sentence(self):
        explanation = "This function is O(n²). It has nested loops."
        assert _extract_verdict(explanation) == "This function is O(n²)"

    def test_extract_verdict_empty_string(self):
        assert _extract_verdict("") == "No verdict produced"

    def test_extract_verdict_single_sentence_no_period(self):
        result = _extract_verdict("This is O(n)")
        assert result == "This is O(n)"


# ── Missing API key ───────────────────────────────────────────────────────────

class TestMissingApiKey:
    def test_raises_environment_error_when_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                run_agent("f.py", "fn")

    def test_error_message_mentions_console(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError, match="console.anthropic.com"):
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

    def test_explanation_is_non_empty_string(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

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

    def test_verdict_is_first_sentence_of_explanation(self):
        with self._mock_client(_two_turn_responses()):
            result = run_agent("f.py", "fn")
        assert result.explanation.startswith(result.verdict)

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
        single_turn = [_response(
            stop_reason="end_turn",
            content=[_text_block("This function is O(1). No loops present.")],
        )]
        with self._mock_client(single_turn):
            result = run_agent("f.py", "fn")
        assert result.verdict == "This function is O(1)"

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
