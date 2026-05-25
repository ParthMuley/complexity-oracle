from __future__ import annotations

import os
import subprocess
import sys

from complexity_oracle.models.analysis import ProfileResult

INPUT_SIZES: list[int] = [10, 100, 1000, 10000]

_RUNNER_TEMPLATE = """\
import sys, time
sys.path.insert(0, {dir_repr})
from {module_name} import {function_name}
data = list(range({n}))
_t0 = time.perf_counter()
{function_name}(data)
_t1 = time.perf_counter()
print((_t1 - _t0) * 1000)
"""


def _build_runner_script(file_path: str, function_name: str, n: int) -> str:
    dir_path = os.path.dirname(os.path.abspath(file_path))
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    return _RUNNER_TEMPLATE.format(
        dir_repr=repr(dir_path),
        module_name=module_name,
        function_name=function_name,
        n=n,
    )


def _is_recursive(file_path: str, function_name: str) -> bool:
    """Return True if *function_name* contains a direct recursive call.

    Imports the parser lazily to avoid a circular dependency at module load time.
    Returns False if the file cannot be parsed (caller will handle the error).
    """
    try:
        from complexity_oracle.core.parser import parse_file  # lazy import
        result = parse_file(file_path)
        return any(
            function_name in msg
            for msg in result.flagged_lines.values()
            if "Recursive call" in msg
        )
    except Exception:  # noqa: BLE001
        return False


def profile_file(
    file_path: str,
    function_name: str,
    timeout_s: float = 5.0,
) -> ProfileResult:
    """Run *function_name* from *file_path* at fixed input sizes and measure runtime.

    The function is called with a single positional argument: list(range(n)).
    Each run executes in an isolated subprocess; the main process is never exposed
    to user code.

    Raises nothing — errors are captured in ProfileResult.error.
    """
    if not os.path.isfile(file_path):
        return ProfileResult(
            input_sizes=[],
            runtimes_ms=[],
            timed_out=False,
            error=f"File not found: {file_path}",
        )

    # Skip profiling for recursive functions — empirical results would be
    # misleading (stack depth depends on input size, not just algorithmic work).
    if _is_recursive(file_path, function_name):
        return ProfileResult(
            input_sizes=[],
            runtimes_ms=[],
            timed_out=False,
            error=(
                "Recursive function — empirical profiling skipped. "
                "Complexity depends on recursion depth and algorithm structure."
            ),
        )

    collected_sizes: list[int] = []
    collected_runtimes: list[float] = []

    for n in INPUT_SIZES:
        script = _build_runner_script(file_path, function_name, n)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(
                input_sizes=collected_sizes,
                runtimes_ms=collected_runtimes,
                timed_out=True,
                error=None,
            )

        if proc.returncode != 0:
            return ProfileResult(
                input_sizes=collected_sizes,
                runtimes_ms=collected_runtimes,
                timed_out=False,
                error=proc.stderr.strip(),
            )

        try:
            ms = float(proc.stdout.strip())
        except ValueError:
            return ProfileResult(
                input_sizes=collected_sizes,
                runtimes_ms=collected_runtimes,
                timed_out=False,
                error=f"Unexpected subprocess output: {proc.stdout!r}",
            )

        collected_sizes.append(n)
        collected_runtimes.append(ms)

    return ProfileResult(
        input_sizes=collected_sizes,
        runtimes_ms=collected_runtimes,
        timed_out=False,
        error=None,
    )
