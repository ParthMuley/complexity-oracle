# Complexity Oracle — Project Context

## What this project is
A CLI tool that detects hidden performance bugs in Python code by combining static AST
analysis with empirical runtime profiling, then using an AI agent to reconcile
disagreements between the two. The innovation is the cross-validation step — most tools
do one or the other, not both.

## Current sprint
**Sprint 1 — Core engine (no LLM calls yet)**
Goal: `oracle analyze myfile.py` works end to end in pure Python.

## Architecture
Single agent, three MCP tools (Sprint 2+). In Sprint 1 we build the tools only.

```
myfile.py → parser → profiler → curve fitter → report builder → CLI output
```

Every stage receives a typed dataclass from the previous. No raw dicts between modules.

## Folder structure
```
complexity_oracle/
  core/
    parser.py       # AST walker → ParseResult
    profiler.py     # code runner → ProfileResult
    fitter.py       # curve fitting → FitResult
    report.py       # assembles → Report
  models/
    analysis.py     # ALL dataclasses live here — single source of truth
  cli/
    main.py         # entry point, no analysis logic
    formatter.py    # all print/display logic lives here only
  tests/
    test_parser.py
    test_profiler.py
    test_fitter.py
```

## Data contracts (models/analysis.py)
```python
class Complexity(Enum):
    O_1, O_LOG_N, O_N, O_N_LOG_N, O_N2, UNKNOWN

@dataclass class UnresolvedCall:  name, line, reason
@dataclass class ParseResult:     functions, max_loop_depth, static_complexity, unresolved_calls, flagged_lines
@dataclass class ProfileResult:   input_sizes, runtimes_ms, timed_out, error
@dataclass class FitResult:       empirical_complexity, r_squared, mismatch, mismatch_reason
@dataclass class Report:          source_file, parse, profile, fit, warnings
```

## Modularity rules — always follow these
- One job per file. parser.py only parses. profiler.py only profiles.
- Modules communicate only through dataclasses from models/analysis.py.
- No print() calls inside core/. Display logic lives in cli/formatter.py only.
- Profiler always runs user code in a sandboxed subprocess with a timeout — never exec in main process.
- No LLM calls in Sprint 1. Agent slot is a commented placeholder in cli/main.py.

## Key design decisions already made
- Input: single Python file (not snippet). User passes the full file path.
- Unresolved external calls: flagged with name, line number, and warning message — not silently ignored.
- Profiler input sizes: n = [10, 100, 1000, 10000] — auto-generated, not user-provided.
- Curve fitting: scipy, fit O(1)/O(log n)/O(n)/O(n log n)/O(n²), pick best by R².
- Mismatch detection: if static_complexity != empirical_complexity, set mismatch=True and explain why.

## What comes in Sprint 2 (don't build yet)
- core/agent.py — ReAct loop using Anthropic tool use API
- mcp/ — MCP server wrapping the three core tools
- Sprint 1 code should require zero changes when Sprint 2 is added

## Tech stack
- Language: Python 3.11+
- AST: stdlib ast module
- Profiling: subprocess + timeit
- Curve fitting: scipy.optimize.curve_fit
- CLI: argparse (Sprint 1), FastAPI added Sprint 3
- Tests: pytest

## Resume context (don't mention in code, useful for naming/framing decisions)
This project targets the Atlassian 2026 New Grad SWE role. Key JD signals:
agentic AI, MCP protocol, function calling, Python, microservices, cloud deployment.
The innovation is static + empirical cross-validation with agent-driven reconciliation.
