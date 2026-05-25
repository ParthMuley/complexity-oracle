# Complexity Oracle — Project Scope & Vision

## The Problem

Engineers write code that looks correct and runs fine in testing — but hidden inside it
are performance traps that only surface at scale. A function that handles 50 users fine
might take 40 seconds with 50,000 users. Current tools are fragmented:

- **Linters** read code structure but frequently misread actual runtime behavior
- **Profilers** measure where time is spent but dump raw numbers with no explanation
- **LLMs** guess at complexity but have no empirical grounding

Nobody combines all three with a layer that cross-validates them and explains the gap
in plain English. That gap is what this project fills.

---

## What We Are Building

A developer productivity tool that takes Python code as input and answers one question:

> *"Why will this code slow down as data grows — and exactly what should I change?"*

It does this by running two independent analyses and comparing them:

1. **Static analysis** — reads the code structure (AST) and predicts complexity
2. **Empirical profiling** — actually runs the code at increasing input sizes and measures it
3. **Cross-validation** — an AI agent reconciles disagreements between the two
4. **Plain-English output** — tells the developer the bottleneck, the line, and the fix

The cross-validation step is the core innovation. A tool that says *"your code looks like
O(n log n) but runs like O(n²) — here's why they disagree"* is more valuable and more
honest than one that guesses from code alone.

---

## Who This Is For

- **Junior developers** shipping features that work in staging but die in production
- **Competitive programmers** debugging TLE (Time Limit Exceeded) submissions
- **Engineering teams** doing code review who want an automated complexity check
- **Anyone** who wants to understand why their code is slow without reading a profiler dump

---

## Current Status

| Sprint | Title | Status |
|---|---|---|
| Sprint 1 | Core engine | ✅ Complete |
| Sprint 2 | Agentic core + MCP | ✅ Complete |
| Sprint 3 | Intelligence layer + API | ✅ Complete |
| Sprint 4 | User experience + deployment | 🔜 Next |

**Tests:** 263 passing · **Last commit:** Sprint 3, Issue #13

---

## Scope by Sprint

### Sprint 1 — Core engine (Weeks 1–2) ✅ COMPLETE
Pure Python pipeline, no LLM. Proves the cross-validation concept works.

**In scope:**
- AST walker that detects loop nesting depth and flags suspicious patterns
- Code runner that executes a function against auto-generated input sizes (n = 10, 100, 1k, 10k)
- Curve fitter (scipy) that fits complexity classes and picks best match by R²
- Mismatch detection when static and empirical results disagree
- CLI: `oracle analyze myfile.py` prints a complexity verdict to terminal
- Unresolved external calls flagged with name, line number, warning message
- Unit tests for all three core modules

**Out of scope:**
- No LLM calls
- No multi-file / folder analysis
- No web UI
- No cloud deployment

**Deliverable:** Running `oracle analyze myfile.py` produces a complexity verdict,
a flagged line, and a list of unresolved external calls.

---

### Sprint 2 — Agentic core + MCP (Weeks 3–4) ✅ COMPLETE
Add the AI reasoning layer. Sprint 1 code requires zero changes.

**In scope:**
- Anthropic tool use API integrated as the orchestrator agent
- ReAct loop: agent calls tools, observes results, reasons across them
- Three Sprint 1 modules refactored as MCP-compliant tool endpoints:
  `analyze_ast`, `run_profiler`, `fit_curve`
- Agent produces structured summary: complexity verdict + mismatch explanation
- One LLM call per analysis (~200 tokens in, ~300 tokens out)

**Out of scope:**
- No multi-agent architecture (one agent, three tools — deliberate choice)
- No web UI yet
- No folder-level analysis yet

**Deliverable:** Agent correctly investigates a mismatch case — where static analysis
and profiler disagree — and explains why in plain English.

---

### Sprint 3 — Intelligence layer + API (Weeks 5–6) ✅ COMPLETE
Make the output smarter and expose it as a service.

**In scope:**
- Plain-English explanation: complexity class + bottleneck line + concrete fix suggestion
  with rewritten code
- Folder mode: `oracle analyze ./src --entry my_function` traverses imports and
  builds a call graph across the codebase
- FastAPI service wrapping the agent — CLI becomes a thin client
- Edge case handling: recursive functions, library calls with hidden complexity,
  hash collision scenarios
- Flag-and-warn behavior for unresolvable cross-file dependencies

**Out of scope:**
- No VS Code extension yet
- No user accounts or persistent storage

**Deliverable:** Tool correctly diagnoses a real-world "works in staging, dies in
production" scenario across multiple files. FastAPI endpoint live locally.

---

### Sprint 4 — User experience + deployment (Weeks 7–9)
Make the tool usable by a stranger who has never seen it before.
Three separate tracks that can be worked in any order.

---

#### Track A — API key onboarding (`oracle setup`)

**Problem:** A first-time user who doesn't have a `.env` file gets a cryptic
`EnvironmentError: ANTHROPIC_API_KEY not set` with no guidance on what to do.

**What to build:**
- New CLI subcommand: `oracle setup`
- Prompts the user to paste their Anthropic API key interactively
- Validates the key by making a cheap test API call (list models or send 1 token)
- Saves the key to `~/.complexity_oracle/.env` (user-level, not project-level)
- On first `oracle analyze` run without a key, prints:
  *"No API key found. Run `oracle setup` to configure one, or use --no-agent to skip AI."*
- `oracle setup --show` prints the currently configured key (masked: `sk-ant-...xxxx`)
- `oracle setup --clear` removes the saved key

**Files to create/modify:**
- `complexity_oracle/cli/setup.py` — new setup wizard
- `complexity_oracle/cli/main.py` — add `setup` subcommand, update key-loading logic
- `tests/test_setup.py`

**Acceptance criteria:**
- A brand-new user can go from zero to first successful analysis in under 2 minutes
- Running `oracle analyze` without a key gives a helpful actionable message, not a stack trace

---

#### Track B — Cloud deployment

**Problem:** The FastAPI service (`api/app.py`) runs locally but is not accessible
to anyone else. There is no live URL to share or put on a resume.

**What to build:**
- `Dockerfile` — containerise the FastAPI service
- `railway.toml` or `render.yaml` — one-click deploy config for Railway or Render (free tier)
- `fly.toml` — alternative for Fly.io
- Environment variable wiring: `ANTHROPIC_API_KEY` set as a secret in the cloud provider
- `GET /` redirect to `/docs` (FastAPI auto-generated Swagger UI)
- Rate limiting middleware (1 request / 10 seconds per IP) — prevents runaway costs
  since each `/analyze` call with agent uses ~1–2 cents of Claude tokens
- Health check endpoint already done (`GET /health`) ✓

**Files to create:**
- `Dockerfile`
- `railway.toml` (or `render.yaml`)
- `complexity_oracle/api/middleware.py` — rate limiter
- `.env.example` — template showing required env vars for deployment

**Acceptance criteria:**
- `docker build . && docker run -p 8000:8000 ...` works locally
- Deployed URL returns 200 on `GET /health`
- Swagger UI accessible at deployed URL `/docs`
- Calling `/analyze` without an API key configured returns 503 with a clear message

---

#### Track C — Multi-language support

**Problem:** The tool only works on Python files. The AST parser, profiler, and
complexity estimator are all Python-specific. A JavaScript or Java file passed to
`oracle analyze` silently fails or produces wrong results.

**What to build (Phase 1 — JavaScript/TypeScript):**
- `complexity_oracle/core/js_parser.py` — uses `tree-sitter` or regex heuristics
  to detect loop nesting depth in `.js` / `.ts` files
- Language detection in `cli/main.py` — check file extension, route to correct parser
- Updated `profiler.py` — JS profiling via `node -e "..."` subprocess (mirrors Python approach)
- `models/analysis.py` — add `language: str` field to `ParseResult`
- Updated `formatter.py` — show language in header

**Scope boundary for Phase 1:**
- Static analysis + loop depth detection only (same as Python Sprint 1)
- No JS-specific agent prompt changes yet
- `.js` and `.ts` only — not Java, Go, etc.
- `node` must be installed on the user's machine (checked at startup, clear error if missing)

**Files to create/modify:**
- `complexity_oracle/core/js_parser.py`
- `complexity_oracle/core/language.py` — language detection + router
- `complexity_oracle/cli/main.py` — language routing
- `tests/test_js_parser.py`

**Acceptance criteria:**
- `oracle analyze myfile.js` produces a static complexity verdict
- `oracle analyze myfile.ts` works the same way
- `oracle analyze myfile.py` behaviour unchanged

---

#### Track D — PyPI packaging + distribution

**Problem:** There is no way for someone to install this tool without cloning the
repo and setting up a venv manually. It cannot be shared or put on a resume as
"installable".

**What to build:**
- Verify `pyproject.toml` is complete for PyPI upload
- `MANIFEST.in` — ensure no dev files or secrets are included in the package
- `README.md` — public-facing, with install instructions, quick-start, and
  architecture diagram (Mermaid or ASCII)
- Test `pip install .` from a clean venv works end-to-end
- Optional: publish to PyPI as `complexity-oracle` or `oracle-complexity`

**Files to create/modify:**
- `README.md` (currently missing or placeholder)
- `MANIFEST.in`
- `pyproject.toml` — verify classifiers, license, author metadata

**Acceptance criteria:**
- `pip install .` in a fresh venv → `oracle analyze demo1_hidden_n2.py --no-agent` works
- No secrets (`.env`, API keys) included in the package

---

**Sprint 4 overall deliverable:**
A tool a stranger can install, configure, and use in under 5 minutes —
with a live deployed URL and support for at least Python and JavaScript.

---

## What Is Deliberately Out of Scope (Full Project)

| Feature | Reason excluded |
|---|---|
| VS Code extension | Increases token spend significantly; CLI is cleaner to demo |
| Multi-agent architecture | One agent + three tools is sufficient; multi-agent adds cost, not value |
| Vertex AI / Gemini | No native MCP support; Anthropic API is the right tool for this |
| Real-time analysis (on keystroke) | Token cost explodes; manual trigger is the correct UX |
| Support for Java / JS | Python first; extensible by design but out of scope for v1 |
| User authentication | Not needed for a portfolio CLI tool |
| Paid infrastructure | Free tiers only (GCP Cloud Run, Railway as backup) |

---

## Architecture Overview

```
User
  │
  ▼
CLI (cli/main.py)
  │
  ▼
Orchestrator Agent  ◄──────────────────────────────┐
  │                                                  │
  ├──► Tool: analyze_ast   (core/parser.py)          │
  │         pure Python · 0 tokens                   │
  │                                                  │
  ├──► Tool: run_profiler  (core/profiler.py)        │
  │         pure Python · 0 tokens                   │
  │                                                  │
  └──► Tool: fit_curve     (core/fitter.py)  ────────┘
            scipy math · 0 tokens

  1 LLM call total (~200 tokens in · ~300 tokens out)
  │
  ▼
Report (models/analysis.py)
  │
  ▼
Formatter (cli/formatter.py)
  │
  ▼
Plain-English output to terminal
```

---

## Token Cost Strategy

LLM sees summaries only, never raw code. Raw code goes to Python tools only.

The agent receives a structured ~200 token summary:
```
Static analysis:  O(n²) — nested loop at line 47
Empirical result: runtime grows as n^2.1 (R²=0.99)
Agreement:        YES
Bottleneck:       list.index() inside for-loop at line 47
Unresolved calls: find_customer() [line 23] — external module
```

Cost per analysis: fractions of a cent. Suitable for a portfolio project and
demonstrably production-conscious for interviews.

---

## Modularity Principles

These apply for the entire project lifetime, not just Sprint 1.

1. **One job per file** — parser.py only parses, profiler.py only profiles
2. **Dataclasses are the contract** — modules talk through models/analysis.py only,
   never raw dicts or tuples
3. **Display logic in one place** — no print() inside core/, everything goes
   through cli/formatter.py
4. **Subprocess sandbox** — user code never runs in the main process
5. **No LLM bleed** — core/ modules have no knowledge of the agent layer;
   agent/ has no knowledge of display logic

---

## Resume Narrative

The story this project tells in an interview:

> "I noticed that static analysis tools and runtime profilers give you different
> information about the same code — and nobody had built something that cross-validates
> them. I built an agentic tool using MCP and Anthropic's tool use API that runs both,
> detects when they disagree, and investigates why. The interesting engineering decision
> was keeping the LLM out of the heavy lifting entirely — all the computation is pure
> Python, and the agent only touches a 200-token summary. That keeps cost near zero
> while the reasoning is still genuinely agentic."

Key signals this hits for Atlassian:
- Agentic AI systems with real tool use (not just API wrapping)
- MCP protocol implementation
- Scale-aware engineering thinking
- Agile build process with incremental delivery
- Python + cloud deployment
- A real problem with a clear before/after

---

## Future Work (post-portfolio)

These are intentionally deferred — good to mention in README and interviews as
evidence of thinking ahead without overbuilding:

- VS Code extension with manual-trigger analysis on selected code block
- Java and JavaScript support (same pipeline, language-specific AST parsers)
- GitHub Action: complexity check on every PR
- Folder-level call graph visualization
- Historical tracking: did this function get slower between commits?
