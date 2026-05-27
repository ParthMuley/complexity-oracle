"""Complexity Oracle — MCP server.

Exposes the three oracle tools over the Model Context Protocol so any MCP
client (Claude Desktop, Claude Code, etc.) can call them directly.

Tools:
  analyze_ast  — static AST analysis → complexity prediction
  run_profiler — empirical runtime profiling at four input sizes
  fit_curve    — scipy curve fitting → complexity class + mismatch detection

── Transports ────────────────────────────────────────────────────────────────

stdio (local — default):
  python -m complexity_oracle.mcp.server

  Register in Claude Desktop / Claude Code (.claude/settings.json):
    {
      "mcpServers": {
        "complexity-oracle": {
          "command": "python",
          "args": ["-m", "complexity_oracle.mcp.server"],
          "cwd": "/path/to/complexity_oracle"
        }
      }
    }

SSE (remote — mounted on FastAPI at /mcp):
  The FastAPI app mounts sse_app() at /mcp.
  Connect from Claude Code with:
    {
      "mcpServers": {
        "complexity-oracle": {
          "type": "sse",
          "url": "https://<cloud-run-url>/mcp"
        }
      }
    }

  Note: run_profiler returns a stub in CLOUD_MODE (profiling disabled for
  security on shared servers). For full empirical profiling use stdio mode
  against a local install.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from complexity_oracle.mcp.tools import (
    _adapter_analyze_ast,
    _adapter_fit_curve,
    _adapter_run_profiler,
)

# ── Server instance ───────────────────────────────────────────────────────────

mcp = FastMCP(
    "complexity-oracle",
    instructions=(
        "You are connected to the Complexity Oracle. "
        "Use analyze_ast first, then run_profiler, then fit_curve "
        "to determine the true algorithmic complexity of a Python function."
    ),
    # Disable DNS rebinding protection — this is a developer tool accessed by
    # MCP clients (Claude Code), not a browser-facing service.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    # Serve the Streamable HTTP endpoint at "/" within the sub-app so that
    # when FastAPI mounts it at "/mcp", the full path is simply "/mcp".
    # (Default would be "/mcp" within the sub-app → "/mcp/mcp" when mounted.)
    streamable_http_path="/",
)


# ── Tool registrations ────────────────────────────────────────────────────────

@mcp.tool(
    name="analyze_ast",
    description=(
        "Parse a Python source file using static AST analysis. "
        "Returns the detected functions, maximum loop nesting depth, "
        "predicted static complexity class, any unresolved external calls, "
        "and flagged lines (e.g. recursion). "
        "Call this first to understand the code structure."
    ),
)
def analyze_ast(file_path: str) -> dict:
    """Statically analyse a Python file and return complexity metadata.

    Args:
        file_path: Absolute or relative path to the Python file to analyse.

    Returns:
        dict with keys: functions, max_loop_depth, static_complexity,
        unresolved_calls, flagged_lines. On error: {"error": "..."}.
    """
    return _adapter_analyze_ast({"file_path": file_path})


@mcp.tool(
    name="run_profiler",
    description=(
        "Profile a specific function by running it at four input sizes "
        "(n = 10, 100, 1 000, 10 000) in isolated subprocesses and measuring "
        "wall-clock time. The function must accept a single argument: list(range(n)). "
        "Call this after analyze_ast to get empirical runtime data."
    ),
)
def run_profiler(
    file_path: str,
    function_name: str,
    timeout_s: float = 5.0,
) -> dict:
    """Profile a Python function at multiple input sizes.

    Args:
        file_path: Path to the Python file containing the function.
        function_name: Name of the function to profile.
        timeout_s: Per-input-size timeout in seconds (default 5.0).

    Returns:
        dict with keys: input_sizes, runtimes_ms, timed_out, error.
    """
    return _adapter_run_profiler(
        {
            "file_path": file_path,
            "function_name": function_name,
            "timeout_s": timeout_s,
        }
    )


@mcp.tool(
    name="fit_curve",
    description=(
        "Fit five complexity models (O(1), O(log n), O(n), O(n log n), O(n²)) "
        "to profiling data using scipy, pick the best fit by R², and check "
        "whether the empirical result matches the static complexity prediction. "
        "Call this after run_profiler, passing the profiler output directly."
    ),
)
def fit_curve(
    input_sizes: list[int],
    runtimes_ms: list[float],
    timed_out: bool,
    static_complexity: str,
) -> dict:
    """Fit a complexity curve to profiler output and detect mismatches.

    Args:
        input_sizes: Input sizes used during profiling, e.g. [10, 100, 1000, 10000].
        runtimes_ms: Measured runtimes in milliseconds, one per input size.
        timed_out: Whether the profiler timed out before completing all sizes.
        static_complexity: Static complexity string from analyze_ast,
            e.g. "O(n)", "O(n²)", "Unknown".

    Returns:
        dict with keys: empirical_complexity, r_squared, mismatch, mismatch_reason.
    """
    return _adapter_fit_curve(
        {
            "input_sizes": input_sizes,
            "runtimes_ms": runtimes_ms,
            "timed_out": timed_out,
            "static_complexity": static_complexity,
        }
    )


# ── ASGI apps for mounting on FastAPI ────────────────────────────────────────

def get_sse_app():
    """Return the FastMCP SSE ASGI app for mounting on FastAPI (legacy)."""
    return mcp.sse_app()


def get_streamable_http_app():
    """Return the FastMCP Streamable HTTP ASGI app for mounting on FastAPI.

    The FastMCP instance is configured with streamable_http_path="/", so
    when this app is mounted at "/mcp" on FastAPI, the MCP endpoint is
    reachable at exactly "/mcp" — the URL Claude Code expects.

    IMPORTANT: call this once at module level in app.py to trigger lazy
    initialization of the session manager, then wire mcp.session_manager.run()
    into FastAPI's lifespan — FastAPI does NOT propagate sub-app lifespans.
    """
    return mcp.streamable_http_app()


# ── Entry point (stdio transport) ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
