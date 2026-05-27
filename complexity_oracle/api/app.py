"""Complexity Oracle — FastAPI service.

Exposes the oracle pipeline as a REST API.

Endpoints:
  GET  /             — browser web UI
  GET  /health        — liveness check
  POST /analyze       — analyse a Python function for complexity

Key resolution for AI agent calls (first match wins):
  1. X-Anthropic-API-Key request header  (per-request, caller-supplied)
  2. ANTHROPIC_API_KEY environment var   (server-wide, owner-supplied)
  3. Neither → HTTP 401

Deployment modes (controlled purely by env vars — same image for both):
  Railway  demo server:   ANTHROPIC_API_KEY set in dashboard → server pays
  Cloud Run public server: no ANTHROPIC_API_KEY set → caller must supply header

Run locally:
  uvicorn complexity_oracle.api.app:app --reload
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from complexity_oracle.api.middleware import RATE_LIMIT, limiter
from complexity_oracle.core.agent import run_agent
from complexity_oracle.mcp.server import get_streamable_http_app, mcp
from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report
from complexity_oracle.models.analysis import Complexity, FitResult, ProfileResult

load_dotenv()
load_dotenv(Path.home() / ".complexity_oracle" / ".env")

# ── MCP Streamable HTTP setup ─────────────────────────────────────────────────
# Build the ASGI app once at import time — this triggers lazy init of
# mcp._session_manager so we can access mcp.session_manager below.
_mcp_asgi_app = get_streamable_http_app()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run the MCP session manager alongside the FastAPI app.

    FastAPI does NOT propagate lifespan events to mounted sub-apps, so we
    must start the session manager here in the top-level lifespan instead.
    """
    async with mcp.session_manager.run():
        yield


# ── Browser UI (inline — no template files needed) ────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complexity Oracle</title>
<style>
:root{
  --bg:#0d1117;--surface:#161b22;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;
  --red:#f85149;--green:#3fb950;--yellow:#e3b341;--orange:#f0883e;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
header{padding:1.25rem 2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.75rem}
header h1{font-size:1.2rem;font-weight:600}
main{max-width:740px;margin:2rem auto;padding:0 1rem;display:flex;flex-direction:column;gap:1rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.25rem;display:flex;flex-direction:column;gap:.85rem}
label{font-size:.82rem;font-weight:500;color:var(--muted);display:block}
.optional{font-weight:400;font-size:.78rem}
input[type=text],input[type=password],textarea{
  background:var(--bg);border:1px solid var(--border);border-radius:6px;
  color:var(--text);padding:.5rem .75rem;font-size:.875rem;width:100%;
  transition:border-color .15s;
}
input:focus,textarea:focus{outline:none;border-color:var(--accent)}
textarea{font-family:'SF Mono','Fira Code','Cascadia Code',monospace;resize:vertical;min-height:210px;line-height:1.5}
.key-wrap{position:relative;margin-top:.4rem}
.key-wrap input{padding-right:2.5rem}
.eye-btn{position:absolute;right:.5rem;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;color:var(--muted);font-size:.95rem;padding:.25rem;line-height:1}
.hint{font-size:.73rem;color:var(--muted)}
.row{display:flex;gap:1rem;align-items:flex-end;flex-wrap:wrap}
.row>div{flex:1;min-width:160px;display:flex;flex-direction:column;gap:.4rem}
.cb-label{display:flex;align-items:center;gap:.5rem;font-size:.85rem;cursor:pointer;white-space:nowrap;padding-bottom:.15rem;color:var(--text)}
.cb-label input{width:auto;accent-color:var(--accent)}
#submit-btn{
  background:var(--accent);color:#0d1117;border:none;border-radius:6px;
  padding:.6rem 1.5rem;font-size:.9rem;font-weight:600;cursor:pointer;
  align-self:flex-start;transition:opacity .15s;
}
#submit-btn:disabled{opacity:.55;cursor:not-allowed}
.error-banner{background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.5);border-radius:8px;padding:1rem 1.25rem;color:var(--red);font-size:.875rem}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.25rem;display:flex;flex-direction:column;gap:1rem}
.result-header{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
.fn-name{font-weight:600;font-size:1rem;font-family:monospace}
.badges{display:flex;gap:.75rem;flex-wrap:wrap}
.badge-group{display:flex;flex-direction:column;align-items:center;gap:.2rem}
.badge-group .lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.badge{border-radius:4px;padding:.2rem .65rem;font-size:.82rem;font-weight:700;font-family:monospace}
.r2{font-size:.75rem;color:var(--muted);margin-left:auto;white-space:nowrap}
.agree-banner{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.35);border-radius:6px;padding:.7rem 1rem;font-size:.85rem;color:var(--green)}
.mismatch-banner{background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.35);border-radius:6px;padding:.7rem 1rem;font-size:.85rem;color:var(--red)}
.cloud-note{font-size:.75rem;color:var(--muted);font-style:italic}
.section-title{font-size:.72rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.agent-grid{display:flex;flex-direction:column;gap:.5rem}
.agent-row{background:var(--bg);border-radius:6px;padding:.75rem;font-size:.875rem;line-height:1.5}
.agent-row strong{display:block;font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.3rem}
pre.snippet{background:var(--bg);border-radius:6px;padding:.75rem;font-size:.8rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all;margin-top:.35rem;border:1px solid var(--border)}
.divider{border:none;border-top:1px solid var(--border)}
.warnings{display:flex;flex-direction:column;gap:.35rem}
.warning-item{font-size:.8rem;color:var(--yellow)}
.spinner{display:inline-block;width:.9em;height:.9em;border:2px solid rgba(13,17,23,.4);border-top-color:#0d1117;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:.4rem}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <span style="font-size:1.4rem">🔬</span>
  <h1>Complexity Oracle</h1>
  <span style="margin-left:auto;font-size:.78rem;color:var(--muted)">Static + Empirical Analysis</span>
</header>
<main>
  <form id="form" class="card">
    <div>
      <label for="key">Anthropic API Key</label>
      <div class="key-wrap">
        <input type="password" id="key" placeholder="sk-ant-api03-…" autocomplete="off" spellcheck="false">
        <button type="button" class="eye-btn" id="toggle-key" title="Show / hide key">👁</button>
      </div>
      <p class="hint" style="margin-top:.4rem">Stored in <code>sessionStorage</code> only — cleared when the tab closes, never sent to our server at rest.</p>
    </div>
    <div>
      <label for="code">Python Code</label>
      <textarea id="code" rows="12" placeholder="def bubble_sort(arr):&#10;    for i in range(len(arr)):&#10;        for j in range(len(arr) - i - 1):&#10;            if arr[j] > arr[j+1]:&#10;                arr[j], arr[j+1] = arr[j+1], arr[j]" style="margin-top:.4rem"></textarea>
    </div>
    <div class="row">
      <div>
        <label for="fn-name">Function name <span class="optional">(optional — auto-detected when one function present)</span></label>
        <input type="text" id="fn-name" placeholder="e.g. bubble_sort">
      </div>
      <label class="cb-label">
        <input type="checkbox" id="skip-ai" checked>
        Skip AI analysis
      </label>
    </div>
    <div>
      <button type="submit" id="submit-btn">Analyze &rarr;</button>
    </div>
  </form>

  <div id="error-banner" hidden></div>
  <div id="result-area" hidden></div>
</main>

<script>
const BADGE = {
  'O(1)':       ['#3fb950','#3fb95022','#3fb95055'],
  'O(log n)':   ['#56d364','#56d36422','#56d36455'],
  'O(n)':       ['#e3b341','#e3b34122','#e3b34155'],
  'O(n log n)': ['#f0883e','#f0883e22','#f0883e55'],
  'O(n\\u00b2)': ['#f85149','#f8514922','#f8514955'],
  'Unknown':    ['#8b949e','#8b949e22','#8b949e55'],
};

function renderBadge(label, text) {
  const [fg, bg, border] = BADGE[text] || BADGE['Unknown'];
  return `<div class="badge-group">
    <span class="lbl">${label}</span>
    <span class="badge" style="color:${fg};background:${bg};border:1px solid ${border}">${text}</span>
  </div>`;
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

const API_BASE = '';

const keyEl = document.getElementById('key');
keyEl.value = sessionStorage.getItem('oracle_key') || '';

document.getElementById('toggle-key').addEventListener('click', () => {
  keyEl.type = keyEl.type === 'password' ? 'text' : 'password';
});

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const key     = keyEl.value.trim();
  const code    = document.getElementById('code').value;
  const fnName  = document.getElementById('fn-name').value.trim();
  const noAgent = document.getElementById('skip-ai').checked;
  const btn     = document.getElementById('submit-btn');
  const errEl   = document.getElementById('error-banner');
  const resEl   = document.getElementById('result-area');

  if (key) sessionStorage.setItem('oracle_key', key);

  // Warn if AI was requested but no key is available
  if (!noAgent && !key) {
    errEl.className = 'error-banner';
    errEl.textContent = 'Paste your Anthropic API key above before enabling AI analysis.';
    errEl.hidden = false;
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analyzing…';
  errEl.hidden = true;
  resEl.hidden = true;

  const body = { code, no_agent: noAgent };
  if (fnName) body.function_name = fnName;

  const headers = { 'Content-Type': 'application/json' };
  if (key) headers['X-Anthropic-API-Key'] = key;

  try {
    const res  = await fetch(`${API_BASE}/analyze`, { method: 'POST', headers, body: JSON.stringify(body) });
    const data = await res.json();

    if (!res.ok) {
      errEl.className = 'error-banner';
      errEl.textContent = data.detail || `Error ${res.status}`;
      errEl.hidden = false;
    } else {
      resEl.innerHTML = buildResult(data);
      resEl.hidden = false;
    }
  } catch (_) {
    errEl.className = 'error-banner';
    errEl.textContent = 'Network error — is the server reachable?';
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Analyze &rarr;';
  }
});

function buildResult(d) {
  const statusHtml = d.mismatch
    ? `<div class="mismatch-banner">⚠️ Mismatch — ${esc(d.mismatch_reason || 'complexities differ')}</div>`
    : `<div class="agree-banner">✅ Static and empirical complexities agree</div>`;

  const cloudHtml = d.profiling_disabled
    ? `<p class="cloud-note">🌐 Cloud mode — empirical profiling is disabled for security. Only static analysis was run.</p>`
    : '';

  let agentHtml = '';
  if (d.agent) {
    const snip = d.agent.code_snippet
      ? `<pre class="snippet">${esc(d.agent.code_snippet)}</pre>`
      : '';
    agentHtml = `
      <hr class="divider">
      <div class="section-title">🤖 Agent Analysis</div>
      <div class="agent-grid">
        <div class="agent-row"><strong>Verdict</strong>${esc(d.agent.verdict)}</div>
        <div class="agent-row"><strong>Why</strong>${esc(d.agent.why)}</div>
        <div class="agent-row"><strong>Fix</strong>${esc(d.agent.fix)}</div>
        ${snip ? `<div class="agent-row"><strong>Code suggestion</strong>${snip}</div>` : ''}
      </div>
      <p class="hint">${d.agent.tokens_used.toLocaleString()} tokens used</p>`;
  } else if (d.agent_error) {
    agentHtml = `
      <hr class="divider">
      <div class="section-title">🤖 Agent Analysis</div>
      <div class="mismatch-banner" style="background:rgba(227,179,65,.1);border-color:rgba(227,179,65,.4);color:var(--yellow)">
        ⚠ Agent failed — ${esc(d.agent_error)}
      </div>`;
  }

  let warningsHtml = '';
  if (d.warnings && d.warnings.length) {
    const items = d.warnings.map(w => `<div class="warning-item">⚠ ${esc(w)}</div>`).join('');
    warningsHtml = `<hr class="divider"><div class="section-title">Warnings</div><div class="warnings">${items}</div>`;
  }

  return `<div class="result-card">
    <div class="result-header">
      <span class="fn-name">${esc(d.function_name)}</span>
      <div class="badges">
        ${renderBadge('Static', d.static_complexity)}
        ${renderBadge('Empirical', d.empirical_complexity)}
      </div>
      <span class="r2">R² = ${d.r_squared.toFixed(3)}</span>
    </div>
    ${statusHtml}
    ${cloudHtml}
    ${agentHtml}
    ${warningsHtml}
  </div>`;
}
</script>
</body>
</html>"""


app = FastAPI(
    title="Complexity Oracle API",
    description=(
        "Detect hidden performance bugs by combining static and empirical analysis.\n\n"
        "**Authentication:** Pass your Anthropic API key via the "
        "`X-Anthropic-API-Key` header, or configure a server-side key."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows the local dev UI (any origin) to POST to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MCP Streamable HTTP transport — mount at /mcp so Claude Code can connect remotely.
# _mcp_asgi_app is built at import time (triggers session manager lazy init).
# The session manager is started in _lifespan above — FastAPI does not propagate
# sub-app lifespans, so we must run it manually there.
app.mount("/mcp", _mcp_asgi_app)


# ── OAuth protected-resource metadata (MCP SDK auth discovery) ───────────────
# The TypeScript MCP SDK (used by Claude Code) proactively fetches this endpoint
# before sending any MCP messages.  Without it FastAPI returns {"detail":"Not Found"}
# which the SDK tries to parse as an OAuth error response → ZodError.
# Returning RFC 9449 metadata with an empty authorization_servers list tells the
# client "this resource exists, no OAuth required" and lets it proceed.
@app.get("/.well-known/oauth-protected-resource", include_in_schema=False)
def oauth_protected_resource_metadata() -> dict:
    return {
        "resource": "https://complexity-oracle-1049073599817.us-central1.run.app/mcp",
        "authorization_servers": [],
    }


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Full Python source code to analyse.")
    function_name: str | None = Field(
        None,
        description="Name of the function to profile. Auto-detected if the file has exactly one function.",
    )
    timeout_s: float = Field(5.0, description="Per-input-size profiling timeout in seconds.")
    no_agent: bool = Field(False, description="Skip the AI agent step when True.")


class AgentOutput(BaseModel):
    verdict: str
    why: str
    fix: str
    code_snippet: str | None
    tokens_used: int


class AnalyzeResponse(BaseModel):
    function_name: str
    static_complexity: str
    empirical_complexity: str
    r_squared: float
    mismatch: bool
    mismatch_reason: str | None
    warnings: list[str]
    profiling_disabled: bool = Field(
        False,
        description="True when running in CLOUD_MODE — profiling is skipped for security.",
    )
    agent: AgentOutput | None
    agent_error: str | None = Field(
        None,
        description="Set when the agent was requested but failed — analysis is still returned.",
    )


# ── Key resolution ────────────────────────────────────────────────────────────

def _resolve_api_key(request: Request) -> str | None:
    """Resolve the Anthropic API key: header → env var → None."""
    return (
        request.headers.get("X-Anthropic-API-Key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or None
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the browser web UI."""
    return HTMLResponse(content=_HTML)


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyse a Python function")
@limiter.limit(RATE_LIMIT)
async def analyze(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse the complexity of a Python function.

    The provided code is written to a temporary file, run through the oracle
    pipeline (parse → [profile → fit] → optional agent), then the temporary file
    is deleted.

    In CLOUD_MODE, profiling is skipped for security — the response will have
    `profiling_disabled: true` and `empirical_complexity: "Unknown"`.

    To use the AI agent, supply your Anthropic key via the `X-Anthropic-API-Key`
    header. If the server has `ANTHROPIC_API_KEY` set, that is used as a fallback.
    """
    cloud_mode = bool(os.environ.get("CLOUD_MODE"))

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(body.code)
            tmp_path = f.name

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            parse = parse_file(tmp_path)
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Syntax error in provided code: {e}")

        # ── Resolve function name ─────────────────────────────────────────────
        function_name = body.function_name
        if function_name is None:
            if not parse.functions:
                raise HTTPException(
                    status_code=400,
                    detail="No functions found in the provided code.",
                )
            if len(parse.functions) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Multiple functions found: {parse.functions}. "
                        "Specify function_name in the request body."
                    ),
                )
            function_name = parse.functions[0]

        # ── Profile + Fit (skipped in CLOUD_MODE) ─────────────────────────────
        if cloud_mode:
            profile = ProfileResult(
                input_sizes=[],
                runtimes_ms=[],
                timed_out=False,
                error="Profiling disabled in cloud mode.",
            )
            fit = FitResult(
                empirical_complexity=Complexity.UNKNOWN,
                r_squared=0.0,
                mismatch=False,
                mismatch_reason=None,
            )
        else:
            profile = profile_file(tmp_path, function_name, timeout_s=body.timeout_s)
            fit = fit_curve(profile, parse)

        # ── Report ─────────────────────────────────────────────────────────────
        report = build_report(tmp_path, parse, profile, fit)

        # ── Agent (optional) ───────────────────────────────────────────────────
        agent_out: AgentOutput | None = None
        agent_error: str | None = None
        if not body.no_agent:
            api_key = _resolve_api_key(request)
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "No Anthropic API key provided. "
                        "Pass your key via the X-Anthropic-API-Key header, "
                        "or run 'oracle setup' to configure a server-side key."
                    ),
                )
            try:
                agent_result = run_agent(
                    tmp_path,
                    function_name,
                    api_key=api_key,
                    cloud_mode=cloud_mode,
                )
                agent_out = AgentOutput(
                    verdict=agent_result.verdict,
                    why=agent_result.why,
                    fix=agent_result.fix,
                    code_snippet=agent_result.code_snippet,
                    tokens_used=agent_result.tokens_used,
                )
            except Exception as e:
                # Agent failure is non-fatal — surface the reason so the UI can show it
                agent_error = str(e)

        # In cloud mode, profiling is intentionally disabled — strip the
        # resulting "Profiling failed" and "Empirical complexity could not be
        # determined" warnings since the UI already shows the cloud-mode banner.
        _CLOUD_NOISE = {"Profiling failed", "Empirical complexity could not be determined"}
        warnings = (
            [w for w in report.warnings if not any(w.startswith(p) for p in _CLOUD_NOISE)]
            if cloud_mode
            else report.warnings
        )

        return AnalyzeResponse(
            function_name=function_name,
            static_complexity=report.parse.static_complexity.value,
            empirical_complexity=report.fit.empirical_complexity.value,
            r_squared=report.fit.r_squared,
            mismatch=report.fit.mismatch,
            mismatch_reason=report.fit.mismatch_reason,
            warnings=warnings,
            profiling_disabled=cloud_mode,
            agent=agent_out,
            agent_error=agent_error,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
