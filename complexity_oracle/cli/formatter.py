from __future__ import annotations

import os
import textwrap

from complexity_oracle.models.analysis import AgentResult, FolderReport, Report

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


# ── Folder report ─────────────────────────────────────────────────────────────

def format_folder_report(folder_report: FolderReport) -> str:
    """Render a FolderReport as a terminal summary table."""
    lines: list[str] = []
    folder_name = os.path.basename(folder_report.folder_path.rstrip("/\\")) or folder_report.folder_path

    # ── Header ────────────────────────────────────────────────────────────────
    lines += ["", _RULE, f"  Complexity Oracle  ·  {folder_name}/", _RULE, ""]

    results = folder_report.results
    n_files = folder_report.total_files
    n_fns = len(results)

    if n_fns == 0:
        lines.append("  No analysable functions found.")
        lines += ["", _RULE, ""]
        return "\n".join(lines)

    lines.append(f"  {n_files} file{'s' if n_files != 1 else ''}  ·  {n_fns} function{'s' if n_fns != 1 else ''} analysed")
    lines.append("")

    # ── Column widths ─────────────────────────────────────────────────────────
    w_file = max(max(len(os.path.basename(r.file_path)) for r in results), 6)
    w_fn   = max(max(len(r.function_name) for r in results), 8)
    w_st   = 10  # "O(n log n)" = 9 chars
    w_emp  = 10
    w_r2   = 6   # "0.999"
    w_verd = 11  # "⚠  MISMATCH"

    # ── Table header ──────────────────────────────────────────────────────────
    hdr = (
        f"  {'File':<{w_file}}  {'Function':<{w_fn}}  "
        f"{'Static':<{w_st}}  {'Empirical':<{w_emp}}  "
        f"{'R²':<{w_r2}}  Verdict"
    )
    sep = "  " + "─" * (len(hdr) - 2)
    lines += [hdr, sep]

    # ── Rows ──────────────────────────────────────────────────────────────────
    mismatches: list[str] = []
    for r in results:
        fname   = os.path.basename(r.file_path)
        verdict = "⚠  MISMATCH" if r.mismatch else "✓  ok"
        if r.error:
            verdict = "✗  error"
        r2_str  = f"{r.r_squared:.3f}" if not r.error else "—"
        row = (
            f"  {fname:<{w_file}}  {r.function_name:<{w_fn}}  "
            f"{r.static_complexity.value:<{w_st}}  {r.empirical_complexity.value:<{w_emp}}  "
            f"{r2_str:<{w_r2}}  {verdict}"
        )
        lines.append(row)
        if r.mismatch:
            mismatches.append(fname)

    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_mismatch = len(mismatches)
    if n_mismatch == 0:
        lines.append(f"  ✓  All {n_fns} function{'s' if n_fns != 1 else ''} look correct.")
    else:
        lines.append(f"  ⚠  {n_mismatch} of {n_fns} function{'s' if n_fns != 1 else ''} {'have' if n_mismatch != 1 else 'has'} a complexity mismatch.")
        for f in mismatches:
            lines.append(f"     Run `oracle analyze {f}` for a full AI analysis.")

    # ── Skipped files ─────────────────────────────────────────────────────────
    if folder_report.skipped_files:
        lines.append("")
        lines.append(f"  Skipped {len(folder_report.skipped_files)} file(s) due to errors:")
        for s in folder_report.skipped_files:
            lines.append(f"    ✗  {s}")

    lines += ["", _RULE, ""]
    return "\n".join(lines)


def print_folder_report(folder_report: FolderReport) -> None:
    """Print a formatted FolderReport to stdout."""
    print(format_folder_report(folder_report))
