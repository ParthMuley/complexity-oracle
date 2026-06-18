# Complexity Oracle

Detect hidden performance bugs in Python code by cross-validating **static analysis**
against **empirical profiling** — then ask an AI agent to explain the gap when they
disagree.

```
def remove_duplicates(items):
    seen = []
    result = []
    for item in items:
        if item not in seen:      # list.__contains__ is O(n) → hidden O(n²)
            seen.append(item)
            result.append(item)
    return result
```

```
$ oracle analyze demo1_hidden_n2.py

════════════════════════════════════════════
  Complexity Oracle  ·  demo1_hidden_n2.py
════════════════════════════════════════════

  Function analysed:  remove_duplicates

  Static analysis:    O(n)
  Empirical result:   O(n²)    [R² = 0.998]

  Verdict:  ⚠   MISMATCH DETECTED

  Mismatch reason:
    Static analysis sees one loop (O(n)), but `item not in seen`
    performs a linear scan of a list on every iteration.

  AI Analysis:

  Verdict:  This function is O(n²) in practice, not O(n).
  Why:
    The single for-loop looks linear, but `seen` is a list and
    `in` on a list is O(n). Doing that inside the loop makes the
    whole function O(n²) — the static analyzer can't see that
    `in` has a cost, it only counts loop nesting.
  Fix:
    Replace `seen` with a set for O(1) membership checks.

  Suggested code:
    def remove_duplicates(items):
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

  [612 tokens used]
════════════════════════════════════════════
```

## Why this exists

Most tools look at code from one angle:

- **Linters** read code structure but misread actual runtime behavior (a list `in`
  check looks just as cheap as a set `in` check to a static analyzer).
- **Profilers** measure runtime but dump raw numbers with no explanation.
- **LLMs** guess at complexity from reading code, with no empirical grounding.

Complexity Oracle runs the static and empirical analyses independently, compares them,
and only calls an AI agent to investigate *when they disagree* — turning "your code
looks O(n log n) but runs like O(n²)" into a plain-English explanation and a concrete
fix, instead of a guess.

## How it works

```
myfile.py → parser (AST) → profiler (subprocess) → curve fitter (scipy) → report
                                                              ↑
                                                  optional: AI agent reconciles
                                                  static vs. empirical results
```

1. **Static analysis** (`core/parser.py`) — walks the AST, measures loop nesting depth,
   predicts a complexity class, and flags any function calls it can't resolve.
2. **Empirical profiling** (`core/profiler.py`) — runs the target function in an
   isolated subprocess at input sizes `n = [10, 100, 1000, 10000]` and times it.
3. **Curve fitting** (`core/fitter.py`) — fits five complexity models
   (`O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`) to the runtime data with
   `scipy.optimize.curve_fit` and picks the best by R².
4. **Cross-validation** — if static and empirical complexity disagree, the result is
   flagged as a mismatch with a reason.
5. **AI reconciliation** (`core/agent.py`, optional) — an Anthropic tool-use agent calls
   the same three analyses as MCP-style tools and produces a verdict, an explanation,
   and a concrete code fix. The agent only ever sees a ~200-token structured summary —
   never the raw source — keeping cost to a fraction of a cent per analysis.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

```bash
# Analyze a single file — auto-detects the function if there's only one
oracle analyze myfile.py

# Specify a function and a profiling timeout
oracle analyze myfile.py --function my_func --timeout 5.0

# Skip the AI step — static + empirical results only, no API key needed
oracle analyze myfile.py --no-agent

# Analyze every function in a folder (optionally recursing into subdirectories)
oracle analyze ./src --recursive

# Configure your Anthropic API key (saved to ~/.complexity_oracle/.env)
oracle setup
oracle setup --show     # print the masked key
oracle setup --clear    # remove the saved key
```

### Running the API / web UI

```bash
uvicorn complexity_oracle.api.app:app --reload
```

Then open `http://localhost:8000` for the browser UI, or `http://localhost:8000/docs`
for the Swagger API docs. `POST /analyze` accepts raw code (not a file path) and an
optional `X-Anthropic-API-Key` header for callers who want to use their own key.

### Running as an MCP server

The same three analysis tools (`analyze_ast`, `run_profiler`, `fit_curve`) are exposed
over the Model Context Protocol, so any MCP client (Claude Code, Claude Desktop) can
call them directly:

```bash
python -m complexity_oracle.mcp.server
```

Register it in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "complexity-oracle": {
      "command": "python",
      "args": ["-m", "complexity_oracle.mcp.server"],
      "cwd": "/path/to/complexity_oracle"
    }
  }
}
```

The FastAPI service also mounts the same MCP server over Streamable HTTP at `/mcp`, for
connecting to a deployed instance remotely.

## Architecture

```
complexity_oracle/
  core/
    parser.py     # AST walker → ParseResult
    profiler.py    # sandboxed subprocess runner → ProfileResult
    fitter.py       # scipy curve fitting → FitResult
    report.py       # assembles a Report from the above
    scanner.py       # folder-mode: discovers files, traverses local imports
    agent.py          # Anthropic tool-use ReAct loop → AgentResult
  models/
    analysis.py       # every dataclass — the single contract between modules
  mcp/
    tools.py            # MCP-tool-shaped adapters over core/
    server.py             # FastMCP server (stdio + Streamable HTTP)
  api/
    app.py                 # FastAPI service, browser UI, rate limiting
  cli/
    main.py                  # argparse entry point
    formatter.py               # all display logic
    setup.py                     # API key setup wizard
tests/                            # one test file per module
```

Modules only communicate through the dataclasses in `models/analysis.py` — never raw
dicts. Display logic lives only in `cli/formatter.py`; `core/` never calls `print()`.
User code is always profiled in a sandboxed subprocess, never `exec`'d in the main
process.

## Testing

```bash
pytest
pytest tests/test_fitter.py -v
```

## Project status

Sprints 1–3 (core engine, agentic core + MCP, intelligence layer + API) are complete.
See `PROJECT_SCOPE.md` for the full sprint breakdown and what's planned for Sprint 4
(onboarding UX, cloud deployment, multi-language support, PyPI packaging).
