from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Complexity(Enum):
    """Algorithmic complexity classes supported by the oracle."""

    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n²)"
    UNKNOWN = "Unknown"


@dataclass
class UnresolvedCall:
    """A function call that cannot be statically analyzed."""

    name: str
    line: int
    reason: str


@dataclass
class ParseResult:
    """Output of the AST parsing stage."""

    functions: list[str]
    max_loop_depth: int
    static_complexity: Complexity
    unresolved_calls: list[UnresolvedCall]
    flagged_lines: dict[int, str]


@dataclass
class ProfileResult:
    """Output of the runtime profiling stage."""

    input_sizes: list[int]
    runtimes_ms: list[float]
    timed_out: bool
    error: str | None


@dataclass
class FitResult:
    """Output of the curve fitting stage."""

    empirical_complexity: Complexity
    r_squared: float
    mismatch: bool
    mismatch_reason: str | None


@dataclass
class Report:
    """Final assembled report combining all analysis stages."""

    source_file: str
    parse: ParseResult
    profile: ProfileResult
    fit: FitResult
    warnings: list[str]


@dataclass
class AgentResult:
    """Output of the Sprint 2 agent reasoning stage."""

    verdict: str               # one-sentence complexity verdict
    why: str                   # explanation of complexity and any mismatch cause
    fix: str                   # concrete developer action to improve performance
    code_snippet: str | None   # optional improved code example
    tokens_used: int           # total tokens consumed across all agent messages
    explanation: str           # raw full response text (kept for debugging)
