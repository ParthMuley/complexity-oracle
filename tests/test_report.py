"""Tests for core/report.py — report assembly and warning generation."""
from __future__ import annotations

import pytest

from complexity_oracle.core.report import build_report
from complexity_oracle.models.analysis import (
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
    Report,
    UnresolvedCall,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

SOURCE = "myfile.py"


def _parse(
    complexity: Complexity = Complexity.O_N,
    unresolved: list[UnresolvedCall] | None = None,
) -> ParseResult:
    return ParseResult(
        functions=["f"],
        max_loop_depth=1,
        static_complexity=complexity,
        unresolved_calls=unresolved or [],
        flagged_lines={},
    )


def _profile(
    timed_out: bool = False,
    error: str | None = None,
) -> ProfileResult:
    return ProfileResult(
        input_sizes=[10, 100, 1000, 10000],
        runtimes_ms=[0.01, 0.1, 1.0, 10.0],
        timed_out=timed_out,
        error=error,
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


def _clean_report() -> Report:
    """A report with no warnings — everything agrees and is clean."""
    return build_report(SOURCE, _parse(), _profile(), _fit())


# ── Assembly ──────────────────────────────────────────────────────────────────

class TestAssembly:
    def test_returns_report_instance(self):
        assert isinstance(_clean_report(), Report)

    def test_source_file_stored(self):
        assert _clean_report().source_file == SOURCE

    def test_source_file_stored_as_given(self):
        r = build_report("/some/path/to/file.py", _parse(), _profile(), _fit())
        assert r.source_file == "/some/path/to/file.py"

    def test_parse_result_preserved(self):
        parse = _parse(Complexity.O_N2)
        r = build_report(SOURCE, parse, _profile(), _fit(Complexity.O_N2))
        assert r.parse is parse

    def test_profile_result_preserved(self):
        profile = _profile()
        r = build_report(SOURCE, _parse(), profile, _fit())
        assert r.profile is profile

    def test_fit_result_preserved(self):
        fit = _fit()
        r = build_report(SOURCE, _parse(), _profile(), fit)
        assert r.fit is fit

    def test_warnings_is_list(self):
        assert isinstance(_clean_report().warnings, list)


# ── No warnings ───────────────────────────────────────────────────────────────

class TestNoWarnings:
    def test_clean_run_has_no_warnings(self):
        assert _clean_report().warnings == []

    def test_high_r2_no_warning(self):
        r = build_report(SOURCE, _parse(), _profile(), _fit(r_squared=0.999))
        assert not any("R²" in w for w in r.warnings)


# ── Individual warnings ───────────────────────────────────────────────────────

class TestWarnings:
    def test_profile_error_fires_warning(self):
        r = build_report(SOURCE, _parse(), _profile(error="subprocess crashed"), _fit())
        assert any("Profiling failed" in w for w in r.warnings)

    def test_profile_error_message_included(self):
        r = build_report(SOURCE, _parse(), _profile(error="subprocess crashed"), _fit())
        assert any("subprocess crashed" in w for w in r.warnings)

    def test_timeout_fires_warning(self):
        r = build_report(SOURCE, _parse(), _profile(timed_out=True), _fit())
        assert any("timed out" in w for w in r.warnings)

    def test_empirical_unknown_fires_warning(self):
        r = build_report(
            SOURCE, _parse(), _profile(),
            _fit(complexity=Complexity.UNKNOWN, r_squared=0.0),
        )
        assert any("Empirical complexity could not be determined" in w for w in r.warnings)

    def test_low_r2_fires_warning(self):
        r = build_report(SOURCE, _parse(), _profile(), _fit(r_squared=0.75))
        assert any("Low R²" in w for w in r.warnings)

    def test_low_r2_value_in_warning(self):
        r = build_report(SOURCE, _parse(), _profile(), _fit(r_squared=0.75))
        assert any("0.75" in w for w in r.warnings)

    def test_high_r2_no_low_r2_warning(self):
        r = build_report(SOURCE, _parse(), _profile(), _fit(r_squared=0.95))
        assert not any("Low R²" in w for w in r.warnings)

    def test_mismatch_fires_warning(self):
        r = build_report(
            SOURCE,
            _parse(Complexity.O_N2),
            _profile(),
            _fit(Complexity.O_N, mismatch=True, mismatch_reason="inner loop fixed"),
        )
        assert any("mismatch" in w.lower() for w in r.warnings)

    def test_mismatch_warning_contains_both_complexities(self):
        r = build_report(
            SOURCE,
            _parse(Complexity.O_N2),
            _profile(),
            _fit(Complexity.O_N, mismatch=True, mismatch_reason="inner loop fixed"),
        )
        mismatch_warnings = [w for w in r.warnings if "mismatch" in w.lower()]
        assert len(mismatch_warnings) == 1
        assert "O(n²)" in mismatch_warnings[0]
        assert "O(n)" in mismatch_warnings[0]

    def test_unresolved_calls_fires_warning(self):
        unresolved = [UnresolvedCall("foo.bar", 5, "External import")]
        r = build_report(SOURCE, _parse(unresolved=unresolved), _profile(), _fit())
        assert any("unresolved" in w.lower() for w in r.warnings)

    def test_unresolved_calls_count_in_warning(self):
        unresolved = [
            UnresolvedCall("foo.bar", 5, "External import"),
            UnresolvedCall("baz.qux", 10, "External import"),
        ]
        r = build_report(SOURCE, _parse(unresolved=unresolved), _profile(), _fit())
        assert any("2" in w for w in r.warnings)

    def test_static_unknown_fires_warning(self):
        r = build_report(
            SOURCE,
            _parse(Complexity.UNKNOWN),
            _profile(),
            _fit(Complexity.UNKNOWN, r_squared=0.0),
        )
        assert any("Static complexity could not be determined" in w for w in r.warnings)

    def test_multiple_warnings_accumulate(self):
        unresolved = [UnresolvedCall("x.y", 1, "External")]
        r = build_report(
            SOURCE,
            _parse(Complexity.O_N2, unresolved=unresolved),
            _profile(timed_out=True),
            _fit(Complexity.O_N, r_squared=0.80, mismatch=True, mismatch_reason="fixed bound"),
        )
        # timeout + low R² + mismatch + unresolved = 4 warnings minimum
        assert len(r.warnings) >= 4

    def test_warnings_are_strings(self):
        unresolved = [UnresolvedCall("x.y", 1, "External")]
        r = build_report(
            SOURCE,
            _parse(unresolved=unresolved),
            _profile(error="boom"),
            _fit(Complexity.UNKNOWN, r_squared=0.0, mismatch=False),
        )
        for w in r.warnings:
            assert isinstance(w, str)
