"""Complexity Oracle — FastAPI service.

Exposes the oracle pipeline as a REST API.

Endpoints:
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
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from complexity_oracle.api.middleware import RATE_LIMIT, limiter
from complexity_oracle.core.agent import run_agent
from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report
from complexity_oracle.models.analysis import Complexity, FitResult, ProfileResult

load_dotenv()
load_dotenv(Path.home() / ".complexity_oracle" / ".env")

app = FastAPI(
    title="Complexity Oracle API",
    description=(
        "Detect hidden performance bugs by combining static and empirical analysis.\n\n"
        "**Authentication:** Pass your Anthropic API key via the "
        "`X-Anthropic-API-Key` header, or configure a server-side key."
    ),
    version="0.1.0",
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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


# ── Key resolution ────────────────────────────────────────────────────────────

def _resolve_api_key(request: Request) -> str | None:
    """Resolve the Anthropic API key: header → env var → None."""
    return (
        request.headers.get("X-Anthropic-API-Key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or None
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
                # Agent failure is non-fatal — return analysis without explanation
                pass

        return AnalyzeResponse(
            function_name=function_name,
            static_complexity=report.parse.static_complexity.value,
            empirical_complexity=report.fit.empirical_complexity.value,
            r_squared=report.fit.r_squared,
            mismatch=report.fit.mismatch,
            mismatch_reason=report.fit.mismatch_reason,
            warnings=report.warnings,
            profiling_disabled=cloud_mode,
            agent=agent_out,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
