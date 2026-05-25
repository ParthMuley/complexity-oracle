from __future__ import annotations

from complexity_oracle.models.analysis import (
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
    Report,
)

# Empirical fits below this R² are flagged as unreliable.
_LOW_R2_THRESHOLD: float = 0.90


def _collect_warnings(
    parse: ParseResult,
    profile: ProfileResult,
    fit: FitResult,
) -> list[str]:
    """Generate pipeline-level warnings from the three analysis results.

    Warnings are additive and ordered: profiling errors first, then fit
    quality, then mismatch, then unresolved calls, then static unknowns.
    """
    warnings: list[str] = []

    # ── Profiling errors ────────────────────────────────────────────────────
    if profile.error is not None:
        warnings.append(f"Profiling failed: {profile.error}")

    if profile.timed_out:
        warnings.append("Profiling timed out — results may be incomplete")

    # ── Empirical fit quality ───────────────────────────────────────────────
    if fit.empirical_complexity is Complexity.UNKNOWN:
        warnings.append("Empirical complexity could not be determined")
    elif fit.r_squared < _LOW_R2_THRESHOLD:
        warnings.append(
            f"Low R² ({fit.r_squared:.2f}) — empirical fit is unreliable; "
            "interpret with caution"
        )

    # ── Mismatch ────────────────────────────────────────────────────────────
    if fit.mismatch:
        warnings.append(
            f"Complexity mismatch: static={parse.static_complexity.value}, "
            f"empirical={fit.empirical_complexity.value}"
        )

    # ── Unresolved external calls ───────────────────────────────────────────
    n_unresolved = len(parse.unresolved_calls)
    if n_unresolved > 0:
        warnings.append(
            f"{n_unresolved} unresolved external call(s) — "
            "complexity may be underestimated"
        )

    # ── Static complexity unknown ───────────────────────────────────────────
    if parse.static_complexity is Complexity.UNKNOWN:
        warnings.append(
            "Static complexity could not be determined "
            "(recursion or dynamic calls detected)"
        )

    return warnings


def build_report(
    source_file: str,
    parse: ParseResult,
    profile: ProfileResult,
    fit: FitResult,
) -> Report:
    """Assemble all analysis stages into a single Report.

    This function performs no analysis — it packages existing results and
    generates pipeline-level warnings for the CLI formatter to display.
    """
    return Report(
        source_file=source_file,
        parse=parse,
        profile=profile,
        fit=fit,
        warnings=_collect_warnings(parse, profile, fit),
    )
