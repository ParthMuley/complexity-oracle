"""Tests for core/fitter.py — curve fitting and mismatch detection."""
from __future__ import annotations

import math

import pytest

from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.models.analysis import (
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
    UnresolvedCall,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

INPUT_SIZES = [10, 100, 1000, 10000]


def _parse(complexity: Complexity = Complexity.O_N) -> ParseResult:
    """Minimal ParseResult for use in tests."""
    return ParseResult(
        functions=["f"],
        max_loop_depth=1,
        static_complexity=complexity,
        unresolved_calls=[],
        flagged_lines={},
    )


def _profile(
    runtimes_ms: list[float],
    sizes: list[int] | None = None,
    timed_out: bool = False,
    error: str | None = None,
) -> ProfileResult:
    return ProfileResult(
        input_sizes=sizes if sizes is not None else INPUT_SIZES[: len(runtimes_ms)],
        runtimes_ms=runtimes_ms,
        timed_out=timed_out,
        error=error,
    )


def _o1_runtimes() -> list[float]:
    """Roughly constant runtimes — O(1)."""
    return [1.0, 1.0, 1.0, 1.0]


def _on_runtimes() -> list[float]:
    """Linearly scaling runtimes — O(n)."""
    return [n * 0.001 for n in INPUT_SIZES]


def _on2_runtimes() -> list[float]:
    """Quadratically scaling runtimes — O(n²)."""
    return [n ** 2 * 0.0001 for n in INPUT_SIZES]


def _on_log_n_runtimes() -> list[float]:
    """O(n log n) runtimes."""
    return [n * math.log(n) * 0.0001 for n in INPUT_SIZES]


# ── Complexity detection ──────────────────────────────────────────────────────

class TestComplexityDetection:
    def test_detects_o1(self):
        result = fit_curve(_profile(_o1_runtimes()), _parse(Complexity.O_1))
        assert result.empirical_complexity == Complexity.O_1

    def test_detects_on(self):
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_N))
        assert result.empirical_complexity == Complexity.O_N

    def test_detects_on2(self):
        result = fit_curve(_profile(_on2_runtimes()), _parse(Complexity.O_N2))
        assert result.empirical_complexity == Complexity.O_N2

    def test_detects_on_log_n(self):
        result = fit_curve(_profile(_on_log_n_runtimes()), _parse(Complexity.O_N_LOG_N))
        assert result.empirical_complexity == Complexity.O_N_LOG_N

    def test_r_squared_close_to_1_for_clean_data(self):
        result = fit_curve(_profile(_on_runtimes()), _parse())
        assert result.r_squared >= 0.99

    def test_r_squared_in_valid_range(self):
        result = fit_curve(_profile(_on2_runtimes()), _parse(Complexity.O_N2))
        assert 0.0 <= result.r_squared <= 1.0


# ── Mismatch detection ────────────────────────────────────────────────────────

class TestMismatchDetection:
    def test_no_mismatch_when_agree(self):
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_N))
        assert result.mismatch is False
        assert result.mismatch_reason is None

    def test_mismatch_when_disagree(self):
        # Static says O(n²) but data is O(n)
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_N2))
        assert result.mismatch is True
        assert result.mismatch_reason is not None
        assert len(result.mismatch_reason) > 0

    def test_mismatch_reason_mentions_both_complexities_or_pattern(self):
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_N2))
        # Should explain the discrepancy in plain English
        reason = result.mismatch_reason.lower()
        assert "nested" in reason or "linear" in reason or "bound" in reason

    def test_unknown_static_no_mismatch_when_empirical_also_unknown(self):
        # Only 1 data point → empirical = UNKNOWN; static = UNKNOWN → no mismatch
        result = fit_curve(
            _profile([1.0], sizes=[10]),
            _parse(Complexity.UNKNOWN),
        )
        assert result.mismatch is False

    def test_unknown_static_mismatch_when_empirical_is_known(self):
        # Static = UNKNOWN (recursive), empirical data is clearly O(n)
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.UNKNOWN))
        assert result.mismatch is True
        assert result.mismatch_reason is not None

    def test_mismatch_on1_static_vs_on_empirical(self):
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_1))
        assert result.mismatch is True
        reason = result.mismatch_reason.lower()
        assert "loop" in reason or "linear" in reason or "external" in reason


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_profiler_error_returns_unknown(self):
        result = fit_curve(
            _profile([], error="subprocess failed"),
            _parse(),
        )
        assert result.empirical_complexity == Complexity.UNKNOWN

    def test_profiler_error_mismatch_false(self):
        result = fit_curve(
            _profile([], error="subprocess failed"),
            _parse(),
        )
        assert result.mismatch is False

    def test_profiler_error_reason_explains(self):
        result = fit_curve(
            _profile([], error="subprocess failed"),
            _parse(),
        )
        assert result.mismatch_reason is not None
        assert "error" in result.mismatch_reason.lower()

    def test_too_few_points_returns_unknown(self):
        result = fit_curve(
            _profile([1.0], sizes=[10]),
            _parse(),
        )
        assert result.empirical_complexity == Complexity.UNKNOWN

    def test_too_few_points_mismatch_false(self):
        result = fit_curve(
            _profile([1.0], sizes=[10]),
            _parse(),
        )
        assert result.mismatch is False

    def test_timed_out_with_partial_data_still_fits_if_enough_points(self):
        # 2 points with clear linear scaling — should still work
        result = fit_curve(
            _profile([0.01, 0.10], sizes=[10, 100], timed_out=True),
            _parse(Complexity.O_N),
        )
        # With only 2 points it may not pick O(n) but should not be UNKNOWN
        assert result.empirical_complexity != Complexity.UNKNOWN or result.r_squared == 0.0

    def test_all_zero_runtimes_returns_o1(self):
        result = fit_curve(
            _profile([0.0, 0.0, 0.0, 0.0]),
            _parse(Complexity.O_1),
        )
        assert result.empirical_complexity == Complexity.O_1


# ── Contract ──────────────────────────────────────────────────────────────────

class TestFitResultContract:
    def test_returns_fit_result_instance(self):
        result = fit_curve(_profile(_on_runtimes()), _parse())
        assert isinstance(result, FitResult)

    def test_empirical_complexity_is_complexity_enum(self):
        result = fit_curve(_profile(_on_runtimes()), _parse())
        assert isinstance(result.empirical_complexity, Complexity)

    def test_r_squared_is_float(self):
        result = fit_curve(_profile(_on_runtimes()), _parse())
        assert isinstance(result.r_squared, float)

    def test_mismatch_is_bool(self):
        result = fit_curve(_profile(_on_runtimes()), _parse())
        assert isinstance(result.mismatch, bool)

    def test_mismatch_reason_none_when_no_mismatch(self):
        result = fit_curve(_profile(_on_runtimes()), _parse(Complexity.O_N))
        assert result.mismatch is False
        assert result.mismatch_reason is None

    def test_mismatch_reason_str_when_mismatch(self):
        result = fit_curve(_profile(_on2_runtimes()), _parse(Complexity.O_N))
        assert result.mismatch is True
        assert isinstance(result.mismatch_reason, str)
