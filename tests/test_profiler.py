from __future__ import annotations

import pytest

from complexity_oracle.core.profiler import INPUT_SIZES, profile_file
from complexity_oracle.models.analysis import ProfileResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LINEAR_CODE = """\
def linear(data):
    total = 0
    for x in data:
        total += x
    return total
"""

_QUADRATIC_CODE = """\
def quadratic(data):
    n = len(data)
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count
"""


def _write(tmp_path, code: str, filename: str = "target.py") -> str:
    p = tmp_path / filename
    p.write_text(code)
    return str(p)


# ---------------------------------------------------------------------------
# TestBasicProfiling
# ---------------------------------------------------------------------------


class TestBasicProfiling:
    def test_linear_function_runs(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert result.error is None
        assert result.timed_out is False
        assert len(result.input_sizes) == 4
        assert len(result.runtimes_ms) == 4

    def test_input_sizes_are_fixed(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert result.input_sizes == [10, 100, 1000, 10000]

    def test_quadratic_slower_than_linear(self, tmp_path):
        lin_path = _write(tmp_path, _LINEAR_CODE, "linear.py")
        quad_path = _write(tmp_path, _QUADRATIC_CODE, "quadratic.py")
        lin = profile_file(lin_path, "linear")
        quad = profile_file(quad_path, "quadratic")
        # compare at n=1000 (index 2) to avoid noise at tiny sizes
        assert quad.runtimes_ms[2] > lin.runtimes_ms[2]

    def test_returns_profile_result_instance(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert isinstance(result, ProfileResult)

    def test_runtimes_non_negative(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert all(t >= 0.0 for t in result.runtimes_ms)


# ---------------------------------------------------------------------------
# TestTimeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_infinite_loop_triggers_timeout(self, tmp_path):
        code = "def hang(data):\n    while True: pass\n"
        path = _write(tmp_path, code)
        result = profile_file(path, "hang", timeout_s=0.5)
        assert result.timed_out is True

    def test_partial_results_on_timeout(self, tmp_path):
        # Fast for small n, very slow for large n
        code = """\
import time
def slow_large(data):
    if len(data) >= 1000:
        time.sleep(60)
"""
        path = _write(tmp_path, code)
        result = profile_file(path, "slow_large", timeout_s=1.0)
        assert result.timed_out is True
        assert len(result.input_sizes) > 0
        assert set(result.input_sizes).issubset(set(INPUT_SIZES))

    def test_timeout_does_not_raise(self, tmp_path):
        code = "def hang(data):\n    while True: pass\n"
        path = _write(tmp_path, code)
        # should return cleanly, not raise
        result = profile_file(path, "hang", timeout_s=0.3)
        assert isinstance(result, ProfileResult)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_bad_file_path(self):
        result = profile_file("/no/such/file.py", "f")
        assert result.error is not None
        assert result.timed_out is False

    def test_bad_function_name(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "nonexistent")
        assert result.error is not None

    def test_function_raises_exception(self, tmp_path):
        code = 'def bad(data):\n    raise ValueError("boom")\n'
        path = _write(tmp_path, code)
        result = profile_file(path, "bad")
        assert result.error is not None

    def test_function_wrong_arity(self, tmp_path):
        code = "def no_args():\n    return 1\n"
        path = _write(tmp_path, code)
        result = profile_file(path, "no_args")
        assert result.error is not None

    def test_error_has_empty_lists_when_first_n_fails(self, tmp_path):
        code = 'def bad(data):\n    raise RuntimeError("fail")\n'
        path = _write(tmp_path, code)
        result = profile_file(path, "bad")
        assert result.input_sizes == []
        assert result.runtimes_ms == []


# ---------------------------------------------------------------------------
# TestRecursionSkip
# ---------------------------------------------------------------------------


class TestRecursionSkip:
    _RECURSIVE_CODE = (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    )

    def test_recursive_function_skipped(self, tmp_path):
        path = _write(tmp_path, self._RECURSIVE_CODE)
        result = profile_file(path, "factorial")
        assert result.error is not None

    def test_recursive_skip_error_mentions_recursion(self, tmp_path):
        path = _write(tmp_path, self._RECURSIVE_CODE)
        result = profile_file(path, "factorial")
        assert "recursive" in result.error.lower()

    def test_recursive_skip_returns_empty_lists(self, tmp_path):
        path = _write(tmp_path, self._RECURSIVE_CODE)
        result = profile_file(path, "factorial")
        assert result.input_sizes == []
        assert result.runtimes_ms == []

    def test_recursive_skip_timed_out_false(self, tmp_path):
        path = _write(tmp_path, self._RECURSIVE_CODE)
        result = profile_file(path, "factorial")
        assert result.timed_out is False

    def test_non_recursive_function_profiled_normally(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert result.error is None
        assert len(result.input_sizes) == 4


# ---------------------------------------------------------------------------
# TestProfileResultContract
# ---------------------------------------------------------------------------


class TestProfileResultContract:
    def test_field_types(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert isinstance(result.input_sizes, list)
        assert isinstance(result.runtimes_ms, list)
        assert all(isinstance(n, int) for n in result.input_sizes)
        assert all(isinstance(t, float) for t in result.runtimes_ms)
        assert isinstance(result.timed_out, bool)
        assert result.error is None or isinstance(result.error, str)

    def test_parallel_lists_same_length(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert len(result.input_sizes) == len(result.runtimes_ms)

    def test_success_error_is_none(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert result.error is None

    def test_success_timed_out_is_false(self, tmp_path):
        path = _write(tmp_path, _LINEAR_CODE)
        result = profile_file(path, "linear")
        assert result.timed_out is False
