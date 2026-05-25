"""Tests for cli/formatter.py — report rendering."""
from __future__ import annotations

import pytest

from complexity_oracle.cli.formatter import format_report
from complexity_oracle.models.analysis import (
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
    Report,
    UnresolvedCall,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

FUNCTION_NAME = "my_func"


def _parse(
    complexity: Complexity = Complexity.O_N,
    unresolved: list[UnresolvedCall] | None = None,
) -> ParseResult:
    return ParseResult(
        functions=[FUNCTION_NAME],
        max_loop_depth=1,
        static_complexity=complexity,
        unresolved_calls=unresolved or [],
        flagged_lines={},
    )


def _profile() -> ProfileResult:
    return ProfileResult(
        input_sizes=[10, 100, 1000, 10000],
        runtimes_ms=[0.01, 0.1, 1.0, 10.0],
        timed_out=False,
        error=None,
    )


def _fit(
    complexity: Complexity = Complexity.O_N,
    r_squared: float = 0.999,
    mismatch: bool = False,
    mismatch_reason: str | None = None,
) -> FitResult:
    return FitResult(
        empirical_complexity=complexity,
        r_squared=r_squared,
        mismatch=mismatch,
        mismatch_reason=mismatch_reason,
    )


def _report(
    source: str = "myfile.py",
    parse: ParseResult | None = None,
    profile: ProfileResult | None = None,
    fit: FitResult | None = None,
    warnings: list[str] | None = None,
) -> Report:
    return Report(
        source_file=source,
        parse=parse or _parse(),
        profile=profile or _profile(),
        fit=fit or _fit(),
        warnings=warnings or [],
    )


def _fmt(report: Report | None = None, fn: str = FUNCTION_NAME) -> str:
    return format_report(report or _report(), fn)


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_string(self):
        assert isinstance(_fmt(), str)

    def test_non_empty(self):
        assert len(_fmt()) > 0


# ── Header ────────────────────────────────────────────────────────────────────

class TestHeader:
    def test_contains_filename(self):
        assert "myfile.py" in _fmt()

    def test_contains_basename_only(self):
        out = format_report(_report(source="/some/deep/path/myfile.py"), FUNCTION_NAME)
        assert "myfile.py" in out
        assert "/some/deep/path/" not in out

    def test_contains_oracle_brand(self):
        assert "Complexity Oracle" in _fmt()


# ── Core fields ───────────────────────────────────────────────────────────────

class TestCoreFields:
    def test_function_name_appears(self):
        assert FUNCTION_NAME in _fmt()

    def test_static_complexity_appears(self):
        out = format_report(_report(parse=_parse(Complexity.O_N2)), FUNCTION_NAME)
        assert "O(n²)" in out

    def test_empirical_complexity_appears(self):
        out = format_report(_report(fit=_fit(Complexity.O_N_LOG_N)), FUNCTION_NAME)
        assert "O(n log n)" in out

    def test_r_squared_appears(self):
        out = format_report(_report(fit=_fit(r_squared=0.987)), FUNCTION_NAME)
        assert "0.987" in out


# ── Verdict ───────────────────────────────────────────────────────────────────

class TestVerdict:
    def test_agreement_when_no_mismatch(self):
        assert "AGREEMENT" in _fmt()

    def test_mismatch_detected_when_mismatch(self):
        out = format_report(
            _report(fit=_fit(mismatch=True, mismatch_reason="fixed bound")),
            FUNCTION_NAME,
        )
        assert "MISMATCH DETECTED" in out

    def test_no_agreement_when_mismatch(self):
        out = format_report(
            _report(fit=_fit(mismatch=True, mismatch_reason="fixed bound")),
            FUNCTION_NAME,
        )
        assert "AGREEMENT" not in out


# ── Mismatch reason ───────────────────────────────────────────────────────────

class TestMismatchReason:
    def test_reason_shown_when_mismatch(self):
        reason = "inner loop has a fixed bound"
        out = format_report(
            _report(fit=_fit(mismatch=True, mismatch_reason=reason)),
            FUNCTION_NAME,
        )
        assert reason in out

    def test_reason_not_shown_when_no_mismatch(self):
        out = _fmt()
        assert "Mismatch reason" not in out


# ── Warnings ──────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_no_warnings_section_when_empty(self):
        assert "Warnings:" not in _fmt()

    def test_warnings_section_appears_when_present(self):
        out = format_report(_report(warnings=["something went wrong"]), FUNCTION_NAME)
        assert "Warnings:" in out

    def test_warning_text_appears(self):
        out = format_report(_report(warnings=["profiling timed out"]), FUNCTION_NAME)
        assert "profiling timed out" in out

    def test_multiple_warnings_all_appear(self):
        out = format_report(
            _report(warnings=["warning one", "warning two", "warning three"]),
            FUNCTION_NAME,
        )
        assert "warning one" in out
        assert "warning two" in out
        assert "warning three" in out


# ── Unresolved calls ──────────────────────────────────────────────────────────

class TestUnresolvedCalls:
    def test_no_unresolved_section_when_empty(self):
        assert "Unresolved calls" not in _fmt()

    def test_unresolved_section_appears_when_present(self):
        calls = [UnresolvedCall("foo.bar", 5, "External import")]
        out = format_report(_report(parse=_parse(unresolved=calls)), FUNCTION_NAME)
        assert "Unresolved calls" in out

    def test_call_name_appears(self):
        calls = [UnresolvedCall("foo.bar", 5, "External import")]
        out = format_report(_report(parse=_parse(unresolved=calls)), FUNCTION_NAME)
        assert "foo.bar" in out

    def test_call_line_appears(self):
        calls = [UnresolvedCall("foo.bar", 42, "External import")]
        out = format_report(_report(parse=_parse(unresolved=calls)), FUNCTION_NAME)
        assert "42" in out

    def test_call_reason_appears(self):
        calls = [UnresolvedCall("foo.bar", 5, "External import (complexity unknown)")]
        out = format_report(_report(parse=_parse(unresolved=calls)), FUNCTION_NAME)
        assert "External import (complexity unknown)" in out

    def test_multiple_calls_all_appear(self):
        calls = [
            UnresolvedCall("foo.bar", 5, "External import"),
            UnresolvedCall("baz.qux", 12, "Undefined function"),
        ]
        out = format_report(_report(parse=_parse(unresolved=calls)), FUNCTION_NAME)
        assert "foo.bar" in out
        assert "baz.qux" in out


# ── Clean run ─────────────────────────────────────────────────────────────────

class TestCleanRun:
    def test_no_empty_sections(self):
        out = _fmt()
        assert "Warnings:" not in out
        assert "Unresolved calls" not in out
        assert "Mismatch reason" not in out
