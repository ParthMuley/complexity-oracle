from __future__ import annotations

import os
import textwrap

from complexity_oracle.models.analysis import AgentResult, Report

_RULE = "═" * 44


def format_report(
    report: Report,
    function_name: str,
    agent_result: AgentResult | None = None,
) -> str:
    """Render a Report (and optional AgentResult) to a human-readable string.

    All display logic for the oracle lives here — no print() calls anywhere
    else in the project.
    """
    lines: list[str] = []

    filename = os.path.basename(report.source_file)

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        "",
        _RULE,
        f"  Complexity Oracle  ·  {filename}",
        _RULE,
        "",
    ]

    # ── Core results ─────────────────────────────────────────────────────────
    lines.append(f"  Function analysed:  {function_name}")
    lines.append("")
    lines.append(f"  Static analysis:    {report.parse.static_complexity.value}")
    lines.append(
        f"  Empirical result:   {report.fit.empirical_complexity.value}"
        f"    [R² = {report.fit.r_squared:.3f}]"
    )
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if report.fit.mismatch:
        lines.append("  Verdict:  ⚠   MISMATCH DETECTED")
    else:
        lines.append("  Verdict:  ✓   AGREEMENT")

    # ── Mismatch reason ───────────────────────────────────────────────────────
    if report.fit.mismatch and report.fit.mismatch_reason:
        lines.append("")
        lines.append("  Mismatch reason:")
        lines.append(f"    {report.fit.mismatch_reason}")

    # ── Warnings ─────────────────────────────────────────────────────────────
    if report.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    ⚠   {w}")

    # ── Unresolved calls ─────────────────────────────────────────────────────
    if report.parse.unresolved_calls:
        lines.append("")
        lines.append("  Unresolved calls:")
        for call in report.parse.unresolved_calls:
            lines.append(f"    Line {call.line:>3}:  {call.name}  — {call.reason}")

    # ── AI Analysis ───────────────────────────────────────────────────────────
    if agent_result is not None:
        lines.append("")
        lines.append("  AI Analysis:")
        lines.append("")

        # Verdict
        lines.append(f"  Verdict:  {agent_result.verdict}")

        # Why
        if agent_result.why:
            lines.append("")
            lines.append("  Why:")
            for ln in textwrap.wrap(agent_result.why, width=68):
                lines.append(f"    {ln}")

        # Fix
        if agent_result.fix:
            lines.append("")
            lines.append("  Fix:")
            for ln in textwrap.wrap(agent_result.fix, width=68):
                lines.append(f"    {ln}")

        # Suggested code snippet
        if agent_result.code_snippet:
            lines.append("")
            lines.append("  Suggested code:")
            for ln in agent_result.code_snippet.splitlines():
                lines.append(f"    {ln}")

        lines.append("")
        lines.append(f"  [{agent_result.tokens_used} tokens used]")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [_RULE, ""]

    return "\n".join(lines)


def print_report(
    report: Report,
    function_name: str,
    agent_result: AgentResult | None = None,
) -> None:
    """Print a formatted Report (and optional AgentResult) to stdout."""
    print(format_report(report, function_name, agent_result))
