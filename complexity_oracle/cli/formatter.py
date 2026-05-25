from __future__ import annotations

import os

from complexity_oracle.models.analysis import Report

_RULE = "═" * 44


def format_report(report: Report, function_name: str) -> str:
    """Render a Report to a human-readable string.

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

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += ["", _RULE, ""]

    return "\n".join(lines)


def print_report(report: Report, function_name: str) -> None:
    """Print a formatted Report to stdout."""
    print(format_report(report, function_name))
