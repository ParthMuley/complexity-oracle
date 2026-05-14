# Complexity Oracle — Session Log

## Session 1 — 2026-05-13

### What was done
**Increment 1 of Sprint 1: Data Models + AST Parser**

#### Files created
| File | Purpose |
|---|---|
| `complexity_oracle/models/analysis.py` | All 6 dataclasses (Complexity enum, UnresolvedCall, ParseResult, ProfileResult, FitResult, Report) — single source of truth for data contracts across the entire pipeline |
| `complexity_oracle/core/parser.py` | AST visitor (`_ComplexityVisitor`) + public `parse_file()` function — walks Python source to extract loop depth, function names, recursion flags, and unresolved calls |
| `tests/test_parser.py` | 24 pytest tests covering: empty files, single/nested/triple loops, comprehensions, recursion, builtins, imports, undefined calls, async functions, error handling, type contracts |
| `pyproject.toml` | Package metadata, Python 3.11+, dev dependencies (pytest, pytest-cov) |
| `.gitignore` | Excludes .venv, __pycache__, .egg-info, .pytest_cache |
| `complexity_oracle/__init__.py` | Package root |
| `complexity_oracle/models/__init__.py` | Re-exports all dataclasses |
| `complexity_oracle/core/__init__.py` | Exports `parse_file` |
| `tests/__init__.py` | Empty (pytest discovery) |

#### Infrastructure set up
- **Virtual environment**: `.venv/` — all dependencies isolated from system Python
- **GitHub repo**: https://github.com/ParthMuley/complexity-oracle (public)
- **GitHub Projects board**: https://github.com/users/ParthMuley/projects/3 ("Sprint 1 — Core Engine")
- **GitHub CLI (`gh`)**: Installed via winget, authenticated as ParthMuley

#### Design decisions and rationale

1. **Dataclasses over dicts** — Type safety, IDE autocomplete, catches typos at import time instead of runtime. Every pipeline stage communicates through these typed contracts.

2. **`ast.NodeVisitor` pattern** — Python's built-in way to walk ASTs. Each node type gets its own `visit_X` method. Cleaner than if/elif chains. Adding new handlers is just adding a method.

3. **Loop depth via enter/exit counters** — `_enter_loop()` increments, `_exit_loop()` decrements, max is tracked. Correctly handles sequential loops (depth stays 1) vs nested (depth goes 2+). Comprehensions count as loops because `[x*2 for x in items]` is O(n) just like a for-loop.

4. **Complexity heuristic is intentionally simple** — depth 0→O(1), depth 1→O(n), depth 2+→O(n²), recursion→UNKNOWN. More sophisticated analysis (O(log n) detection, divide-and-conquer patterns) is deferred to Sprint 2 where the AI agent can reason about it.

5. **Call classification uses `dir(builtins)`** — Automatically includes all ~150 Python builtins without hardcoding. Imported names are tracked and flagged as "external" since we can't know their complexity statically. Undefined names are flagged separately.

6. **Errors propagate naturally** — `FileNotFoundError` and `SyntaxError` bubble up to the caller. No wrapping — the CLI layer (Sprint 1, Issue #5) will handle user-facing error messages.

7. **All models defined upfront** — Even though only `ParseResult` and `UnresolvedCall` are used now, all 6 dataclasses are in `models/analysis.py`. This avoids circular imports later and establishes the full contract early.

### Known limitations (accepted)
- Mutual recursion not detected (only direct self-calls)
- Complexity caps at O(n²)
- Dynamic calls (`getattr`, `eval`) not resolved
- No class hierarchy / method override tracking

### Test results
```
24 passed in 0.15s
```

### What's next (Sprint 1 remaining)
| Issue | Status |
|---|---|
| #2 Profiler: subprocess + timeit | Todo |
| #3 Curve fitter: scipy best-fit | Todo |
| #4 Report builder | Todo |
| #5 CLI: argparse + formatter | Todo |

### How to resume
```bash
# Activate the virtual environment
source D:/projects/Atlassian/complexity_oracle/.venv/Scripts/activate

# Run existing tests to verify everything works
pytest tests/test_parser.py -v

# Next task: implement core/profiler.py (Issue #2)
```
