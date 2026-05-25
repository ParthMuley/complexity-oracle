"""Complexity Oracle agent — ReAct loop using Anthropic tool use API.

The agent receives a file path and function name, calls the three MCP tools
(analyze_ast → run_profiler → fit_curve) in a ReAct loop, then produces a
structured verdict explaining the true complexity and any mismatch.

Sprint 2 scope: one agent, three tools, one LLM call cycle (~200 in / ~300 out tokens).
Sprint 3 will add the MCP server wrapper around these tools.
"""
from __future__ import annotations

import os
import re

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

After all three tool calls, respond using EXACTLY this format. Use plain text only —
no markdown, no **, no ###, no bullet points.

VERDICT: <one sentence — state the true complexity class and whether it matches static analysis>
WHY: <2-3 sentences — explain what causes this complexity; if there is a mismatch, name the exact line or operation responsible>
FIX: <one sentence — concrete action the developer should take>
CODE:
<optional: 3-10 lines of improved Python code showing the fix; omit this entire section if no code change is needed>

Rules:
- VERDICT must start with the complexity class, e.g. "This function is O(n²) in practice".
- WHY must name the specific line number or operation causing the cost.
- If static and empirical complexity agree and the code is fine, FIX should say "No change needed.".
- Only include CODE if you are showing a concrete code improvement.
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

        # ── Agent finished — parse structured response ────────────────────────
        if response.stop_reason == "end_turn":
            raw = _extract_text(response.content)
            sections = _parse_structured_response(raw)
            return AgentResult(
                verdict=sections["verdict"],
                why=sections["why"],
                fix=sections["fix"],
                code_snippet=sections["code_snippet"],
                tokens_used=total_tokens,
                explanation=raw,
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
        why="Agent did not produce a final explanation within the turn limit.",
        fix="Re-run the analysis or increase the turn limit.",
        code_snippet=None,
        tokens_used=total_tokens,
        explanation="",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(content: list) -> str:
    """Pull the first text block out of a message content list."""
    for block in content:
        if hasattr(block, "text") and block.text:
            return block.text.strip()
    return ""


def _parse_structured_response(text: str) -> dict:
    """Parse VERDICT/WHY/FIX/CODE sections from a structured agent response.

    Returns a dict with keys: verdict, why, fix, code_snippet.
    Falls back gracefully if the model didn't follow the format exactly.
    """
    _LABELS = ["VERDICT", "WHY", "FIX", "CODE"]
    pattern = re.compile(
        r"^(" + "|".join(_LABELS) + r"):\s*", re.MULTILINE
    )
    parts = pattern.split(text)
    # parts = [pre_text, LABEL1, content1, LABEL2, content2, ...]

    parsed: dict[str, str | None] = {k: None for k in _LABELS}
    i = 1
    while i + 1 < len(parts):
        label = parts[i]
        content = parts[i + 1].strip()
        if label in parsed:
            parsed[label] = content or None
        i += 2

    # Graceful fallback — if structured parsing failed, use raw text
    if not parsed["VERDICT"]:
        first_sentence = text.split(".")[0].strip()
        parsed["VERDICT"] = first_sentence if first_sentence else "No verdict produced"
    if not parsed["WHY"]:
        parsed["WHY"] = text  # surface the full text so nothing is lost

    return {
        "verdict": parsed["VERDICT"] or "No verdict produced",
        "why": parsed["WHY"] or "",
        "fix": parsed["FIX"] or "No fix suggested.",
        "code_snippet": parsed["CODE"],
    }
