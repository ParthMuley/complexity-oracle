"""Complexity Oracle — FastAPI service.

Exposes the oracle pipeline as a REST API.

Endpoints:
  GET  /health        — liveness check
  POST /analyze       — analyse a Python function for complexity

Run locally:
  uvicorn complexity_oracle.api.app:app --reload
"""
from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from complexity_oracle.core.agent import run_agent
from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report

load_dotenv()

app = FastAPI(
    title="Complexity Oracle API",
    description="Detect hidden performance bugs by combining static and empirical analysis.",
    version="0.1.0",
)


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
    agent: AgentOutput | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyse a Python function")
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse the complexity of a Python function.

    The provided code is written to a temporary file, run through the oracle
    pipeline (parse → profile → fit → optional agent), then the temporary file
    is deleted.  The response contains both static and empirical complexity
    results plus an optional AI-generated explanation.
    """
    # ── Write code to a temp file ─────────────────────────────────────────────
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(request.code)
            tmp_path = f.name

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            parse = parse_file(tmp_path)
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Syntax error in provided code: {e}")

        # ── Resolve function name ─────────────────────────────────────────────
        function_name = request.function_name
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

        # ── Profile → Fit → Report ─────────────────────────────────────────────
        profile = profile_file(tmp_path, function_name, timeout_s=request.timeout_s)
        fit = fit_curve(profile, parse)
        report = build_report(tmp_path, parse, profile, fit)

        # ── Agent (optional) ───────────────────────────────────────────────────
        agent_out: AgentOutput | None = None
        if not request.no_agent:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "ANTHROPIC_API_KEY is not set. "
                        "Set the environment variable or pass no_agent=true to skip the agent."
                    ),
                )
            agent_result = run_agent(tmp_path, function_name)
            agent_out = AgentOutput(
                verdict=agent_result.verdict,
                why=agent_result.why,
                fix=agent_result.fix,
                code_snippet=agent_result.code_snippet,
                tokens_used=agent_result.tokens_used,
            )

        return AnalyzeResponse(
            function_name=function_name,
            static_complexity=report.parse.static_complexity.value,
            empirical_complexity=report.fit.empirical_complexity.value,
            r_squared=report.fit.r_squared,
            mismatch=report.fit.mismatch,
            mismatch_reason=report.fit.mismatch_reason,
            warnings=report.warnings,
            agent=agent_out,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
