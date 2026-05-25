from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

from complexity_oracle.models.analysis import (
    Complexity,
    FitResult,
    ParseResult,
    ProfileResult,
)

# Minimum data points needed for a meaningful curve fit.
_MIN_POINTS: int = 2

# ── Model functions ──────────────────────────────────────────────────────────
# Each takes a numpy array of input sizes and one or more free parameters.
# scipy.optimize.curve_fit finds the best (a, b) for each model, then we
# compute R² to pick the winner.

def _model_o1(n: np.ndarray, a: float) -> np.ndarray:         # O(1)
    return np.full_like(n, a, dtype=float)

def _model_log_n(n: np.ndarray, a: float, b: float) -> np.ndarray:   # O(log n)
    return a * np.log(n) + b

def _model_n(n: np.ndarray, a: float, b: float) -> np.ndarray:       # O(n)
    return a * n + b

def _model_n_log_n(n: np.ndarray, a: float, b: float) -> np.ndarray: # O(n log n)
    return a * n * np.log(n) + b

def _model_n2(n: np.ndarray, a: float, b: float) -> np.ndarray:      # O(n²)
    return a * n ** 2 + b


# Ordered list of (Complexity class, model function) — order doesn't affect
# correctness but prefer simpler models for readability in ties.
_MODELS: list[tuple[Complexity, Callable]] = [
    (Complexity.O_1,       _model_o1),
    (Complexity.O_LOG_N,   _model_log_n),
    (Complexity.O_N,       _model_n),
    (Complexity.O_N_LOG_N, _model_n_log_n),
    (Complexity.O_N2,      _model_n2),
]

# ── Mismatch reason templates ────────────────────────────────────────────────

_MISMATCH_REASONS: dict[tuple[Complexity, Complexity], str] = {
    (Complexity.O_N2,      Complexity.O_N):       (
        "Nested loops found statically but runtime is linear — "
        "inner loop likely has a fixed bound"
    ),
    (Complexity.O_N2,      Complexity.O_1):       (
        "Nested loops found statically but runtime is constant — "
        "loops may iterate over fixed-size data"
    ),
    (Complexity.O_N2,      Complexity.O_LOG_N):   (
        "Nested loops found statically but runtime is logarithmic — "
        "inner loop likely has a logarithmically shrinking bound"
    ),
    (Complexity.O_N2,      Complexity.O_N_LOG_N): (
        "Nested loops found statically but runtime is O(n log n) — "
        "inner loop bound may grow sub-linearly"
    ),
    (Complexity.O_N,       Complexity.O_N2):      (
        "Single loop found statically but runtime is quadratic — "
        "loop body likely calls an O(n) function (e.g. list.index, 'in' on a list)"
    ),
    (Complexity.O_N,       Complexity.O_1):       (
        "Single loop found statically but runtime is constant — "
        "loop may iterate over a fixed-size collection regardless of input"
    ),
    (Complexity.O_N,       Complexity.O_LOG_N):   (
        "Single loop found statically but runtime is logarithmic — "
        "loop bound may depend on log(n) rather than n"
    ),
    (Complexity.O_N,       Complexity.O_N_LOG_N): (
        "Single loop found statically but runtime is O(n log n) — "
        "loop body may contain a hidden O(log n) operation"
    ),
    (Complexity.O_1,       Complexity.O_N):       (
        "No loops detected statically but runtime grows linearly — "
        "hidden complexity likely inside an external or built-in call"
    ),
    (Complexity.O_1,       Complexity.O_N2):      (
        "No loops detected statically but runtime is quadratic — "
        "quadratic work is hidden inside an external or built-in call"
    ),
    (Complexity.O_1,       Complexity.O_LOG_N):   (
        "No loops detected statically but runtime grows logarithmically — "
        "hidden complexity inside an external call (e.g. binary search)"
    ),
    (Complexity.O_1,       Complexity.O_N_LOG_N): (
        "No loops detected statically but runtime is O(n log n) — "
        "likely a sort or similar hidden inside an external call"
    ),
    (Complexity.UNKNOWN,   Complexity.O_N):       (
        "Recursion detected but empirical fit is O(n) — "
        "likely tail-recursive or bounded depth"
    ),
    (Complexity.UNKNOWN,   Complexity.O_N2):      (
        "Recursion detected but empirical fit is O(n²) — "
        "recursive branching may be quadratic in practice"
    ),
    (Complexity.UNKNOWN,   Complexity.O_LOG_N):   (
        "Recursion detected but empirical fit is O(log n) — "
        "looks like a divide-and-conquer algorithm with O(1) work per level"
    ),
    (Complexity.UNKNOWN,   Complexity.O_N_LOG_N): (
        "Recursion detected but empirical fit is O(n log n) — "
        "consistent with a divide-and-conquer algorithm (e.g. merge sort)"
    ),
    (Complexity.UNKNOWN,   Complexity.O_1):       (
        "Recursion detected but runtime is constant — "
        "recursion depth may be bounded by a fixed value"
    ),
}

def _mismatch_reason(static: Complexity, empirical: Complexity) -> str:
    """Return a human-readable explanation for a static vs empirical mismatch."""
    key = (static, empirical)
    if key in _MISMATCH_REASONS:
        return _MISMATCH_REASONS[key]
    return (
        f"Static analysis predicted {static.value} but empirical fit is "
        f"{empirical.value} — investigate external calls or hidden data structure costs"
    )


# ── R² computation ───────────────────────────────────────────────────────────

def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Coefficient of determination. Clamped to [0, 1]."""
    ss_res = float(np.sum((y_actual - y_predicted) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    if ss_tot == 0.0:
        # All runtimes identical — model is perfect if it also predicts a constant.
        return 1.0 if ss_res == 0.0 else 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


# ── Public API ───────────────────────────────────────────────────────────────

def fit_curve(profile: ProfileResult, parse: ParseResult) -> FitResult:
    """Fit complexity curves to profiling data and detect mismatches.

    Returns FitResult with empirical_complexity=UNKNOWN when:
    - profile.error is set (profiling failed)
    - fewer than _MIN_POINTS data points are available
    - all curve_fit calls fail to converge
    """
    # ── Guard: profiling failed ──────────────────────────────────────────────
    if profile.error is not None:
        return FitResult(
            empirical_complexity=Complexity.UNKNOWN,
            r_squared=0.0,
            mismatch=False,
            mismatch_reason="Profiling error — empirical complexity could not be determined",
        )

    # ── Guard: too few points ────────────────────────────────────────────────
    n_points = len(profile.input_sizes)
    if n_points < _MIN_POINTS:
        return FitResult(
            empirical_complexity=Complexity.UNKNOWN,
            r_squared=0.0,
            mismatch=False,
            mismatch_reason="Too few data points — profiling may have timed out early",
        )

    x = np.array(profile.input_sizes, dtype=float)
    y = np.array(profile.runtimes_ms, dtype=float)

    best_complexity = Complexity.UNKNOWN
    best_r2 = -1.0

    for complexity, model_fn in _MODELS:
        try:
            popt, _ = curve_fit(model_fn, x, y, maxfev=10_000)
            y_pred = model_fn(x, *popt)
            r2 = _r_squared(y, y_pred)
        except (RuntimeError, ValueError):
            # curve_fit failed to converge or got invalid input — skip this model.
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_complexity = complexity

    # ── Guard: all models failed ─────────────────────────────────────────────
    if best_complexity is Complexity.UNKNOWN:
        return FitResult(
            empirical_complexity=Complexity.UNKNOWN,
            r_squared=0.0,
            mismatch=False,
            mismatch_reason="Curve fitting failed — all models failed to converge",
        )

    # ── Mismatch detection ───────────────────────────────────────────────────
    static = parse.static_complexity
    mismatch = static != best_complexity and static is not Complexity.UNKNOWN or (
        static is Complexity.UNKNOWN and best_complexity is not Complexity.UNKNOWN
    )

    # Re-evaluate: UNKNOWN static + UNKNOWN empirical → no mismatch
    if static is Complexity.UNKNOWN and best_complexity is Complexity.UNKNOWN:
        mismatch = False

    reason: str | None = None
    if mismatch:
        reason = _mismatch_reason(static, best_complexity)

    return FitResult(
        empirical_complexity=best_complexity,
        r_squared=round(best_r2, 4),
        mismatch=mismatch,
        mismatch_reason=reason,
    )
