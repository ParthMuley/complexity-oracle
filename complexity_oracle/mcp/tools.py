"""MCP tool definitions and JSON adapters for the Complexity Oracle agent.

Each tool wraps a Sprint 1 core function with:
  - A JSON schema (Anthropic tool_use format) for the agent to reason with
  - An adapter function that accepts a JSON dict and returns a JSON dict

The agent calls these via dispatch_tool(name, inputs) — it never touches
the core functions directly.
"""
from __future__ import annotations

import json

from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.fitter import fit_curve as _fit_curve
from complexity_oracle.models.analysis import (
    Complexity,
    ParseResult,
    ProfileResult,
)

# ── Tool definitions (Anthropic tool_use schema format) ──────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "analyze_ast",
        "description": (
            "Parse a Python source file using static AST analysis. "
            "Returns the detected functions, maximum loop nesting depth, "
            "predicted static complexity class, any unresolved external calls, "
            "and flagged lines (e.g. recursion). "
            "Call this first to understand the code structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the Python file to analyse.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "run_profiler",
        "description": (
            "Profile a specific function by running it at four input sizes "
            "(n = 10, 100, 1 000, 10 000) in isolated subprocesses and measuring "
            "wall-clock time. The function must accept a single argument: list(range(n)). "
            "Call this after analyze_ast to get empirical runtime data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file containing the function.",
                },
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to profile.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Per-input-size timeout in seconds. Default 5.0.",
                },
            },
            "required": ["file_path", "function_name"],
        },
    },
    {
        "name": "fit_curve",
        "description": (
            "Fit five complexity models (O(1), O(log n), O(n), O(n log n), O(n²)) "
            "to profiling data using scipy, pick the best fit by R², and check "
            "whether the empirical result matches the static complexity prediction. "
            "Call this after run_profiler, passing the profiler output directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input_sizes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Input sizes used during profiling (e.g. [10, 100, 1000, 10000]).",
                },
                "runtimes_ms": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Measured runtimes in milliseconds, one per input size.",
                },
                "timed_out": {
                    "type": "boolean",
                    "description": "Whether the profiler timed out before completing all sizes.",
                },
                "static_complexity": {
                    "type": "string",
                    "description": (
                        "Static complexity from analyze_ast "
                        "(e.g. 'O(n)', 'O(n²)', 'Unknown')."
                    ),
                },
            },
            "required": ["input_sizes", "runtimes_ms", "timed_out", "static_complexity"],
        },
    },
]

# ── Adapter functions ─────────────────────────────────────────────────────────
# Each adapter: dict → dict (always JSON-serialisable, never raises).

def _adapter_analyze_ast(inputs: dict) -> dict:
    file_path: str = inputs["file_path"]
    try:
        result = parse_file(file_path)
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except SyntaxError as e:
        return {"error": f"Syntax error in {file_path}: {e}"}

    return {
        "functions": result.functions,
        "max_loop_depth": result.max_loop_depth,
        "static_complexity": result.static_complexity.value,
        "unresolved_calls": [
            {"name": c.name, "line": c.line, "reason": c.reason}
            for c in result.unresolved_calls
        ],
        "flagged_lines": {str(line): msg for line, msg in result.flagged_lines.items()},
    }


def _adapter_run_profiler(inputs: dict) -> dict:
    file_path: str = inputs["file_path"]
    function_name: str = inputs["function_name"]
    timeout_s: float = float(inputs.get("timeout_s", 5.0))

    result = profile_file(file_path, function_name, timeout_s=timeout_s)

    return {
        "input_sizes": result.input_sizes,
        "runtimes_ms": result.runtimes_ms,
        "timed_out": result.timed_out,
        "error": result.error,
    }


def _adapter_fit_curve(inputs: dict) -> dict:
    # Reconstruct minimal dataclasses from the agent-supplied JSON.
    try:
        static_str: str = inputs["static_complexity"]
        # Map the string value back to the Complexity enum.
        static_complexity = next(
            (c for c in Complexity if c.value == static_str),
            Complexity.UNKNOWN,
        )

        profile = ProfileResult(
            input_sizes=inputs["input_sizes"],
            runtimes_ms=inputs["runtimes_ms"],
            timed_out=inputs["timed_out"],
            error=None,
        )
        parse = ParseResult(
            functions=[],
            max_loop_depth=0,
            static_complexity=static_complexity,
            unresolved_calls=[],
            flagged_lines={},
        )
    except (KeyError, TypeError) as e:
        return {"error": f"Invalid fit_curve input: {e}"}

    result = _fit_curve(profile, parse)

    return {
        "empirical_complexity": result.empirical_complexity.value,
        "r_squared": result.r_squared,
        "mismatch": result.mismatch,
        "mismatch_reason": result.mismatch_reason,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_ADAPTERS: dict[str, callable] = {
    "analyze_ast": _adapter_analyze_ast,
    "run_profiler": _adapter_run_profiler,
    "fit_curve": _adapter_fit_curve,
}


def dispatch_tool(tool_name: str, inputs: dict) -> str:
    """Route a tool call by name, run the adapter, return JSON string.

    Always returns a JSON string — never raises. Unknown tool names return
    an error JSON so the agent can handle them gracefully.
    """
    adapter = _ADAPTERS.get(tool_name)
    if adapter is None:
        return json.dumps({"error": f"Unknown tool: {tool_name!r}"})
    result = adapter(inputs)
    return json.dumps(result)
