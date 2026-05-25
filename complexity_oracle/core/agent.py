"""Complexity Oracle agent — ReAct loop using Anthropic tool use API.

The agent receives a file path and function name, calls the three MCP tools
(analyze_ast → run_profiler → fit_curve) in a ReAct loop, then produces a
plain-English verdict explaining the true complexity and any mismatch.

Sprint 2 scope: one agent, three tools, one LLM call cycle (~200 in / ~300 out tokens).
Sprint 3 will add the MCP server wrapper around these tools.
"""
from __future__ import annotations

import os

import anthropic

from complexity_oracle.mcp.tools import TOOL_DEFINITIONS, dispatch_tool
from complexity_oracle.models.analysis import AgentResult

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_TURNS = 10  # safety cap — prevents runaway loops

SYSTEM_PROMPT = """\
You are the Complexity Oracle investigator. Your job is to determine the true
algorithmic complexity of a Python function by combining static code analysis
with empirical runtime profiling.

Use your tools in this order:
1. analyze_ast   — understand the code structure and get the static complexity prediction
2. run_profiler  — measure actual runtime at four input sizes (n = 10, 100, 1000, 10000)
3. fit_curve     — pass the profiler output to determine the empirical complexity class

After all three tool calls, deliver a concise plain-English verdict that covers:
- The true (empirical) complexity class
- Whether static analysis agrees with the empirical result
- If there is a mismatch: exactly WHY they disagree (what hidden cost causes it)
- A concrete suggestion for what the developer should change to fix it

Be specific. Name the line or operation causing the hidden cost. Keep the explanation
under 150 words. Start your first sentence with the verdict (e.g. "This function is O(n²)
in practice...").
"""


def run_agent(file_path: str, function_name: str) -> AgentResult:
    """Run the oracle agent on a single function.

    Raises EnvironmentError if ANTHROPIC_API_KEY is not set.
    Raises anthropic.APIError on API failures.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Get a key at console.anthropic.com and set it with:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyse the function `{function_name}` in the file `{file_path}`. "
                "Use your tools to determine its true complexity and explain any mismatch."
            ),
        }
    ]

    total_tokens = 0

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # ── Agent finished — extract explanation ─────────────────────────────
        if response.stop_reason == "end_turn":
            explanation = _extract_text(response.content)
            verdict = _extract_verdict(explanation)
            return AgentResult(
                verdict=verdict,
                explanation=explanation,
                tokens_used=total_tokens,
            )

        # ── Agent wants to call tools ────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Record the assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_json = dispatch_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — break and return what we have
        break

    # Fallback if loop exhausted without end_turn
    return AgentResult(
        verdict="Analysis incomplete",
        explanation="Agent did not produce a final explanation within the turn limit.",
        tokens_used=total_tokens,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(content: list) -> str:
    """Pull the first text block out of a message content list."""
    for block in content:
        if hasattr(block, "text") and block.text:
            return block.text.strip()
    return ""


def _extract_verdict(explanation: str) -> str:
    """Return the first sentence of the explanation as the verdict."""
    if not explanation:
        return "No verdict produced"
    first = explanation.split(".")[0].strip()
    return first if first else explanation[:120]
