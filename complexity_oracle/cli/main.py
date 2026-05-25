from __future__ import annotations

import argparse
import io
import sys

# Ensure stdout can handle Unicode (e.g. ═, ⚠, ✓) on Windows terminals.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report
from complexity_oracle.cli.formatter import print_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle",
        description="Detect hidden performance bugs by combining static and empirical analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a Python file for complexity.")
    analyze.add_argument("file", help="Path to the Python file to analyze.")
    analyze.add_argument(
        "--function",
        default=None,
        help="Name of the function to profile. Auto-detected if the file contains exactly one function.",
    )
    analyze.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-input-size profiling timeout in seconds (default: 5.0).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        _run_analyze(args.file, args.function, args.timeout)


def _run_analyze(file_path: str, function_name: str | None, timeout_s: float) -> None:
    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        parse = parse_file(file_path)
    except FileNotFoundError:
        print(f"Error: file not found — {file_path}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"Error: syntax error — {e}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve function name ────────────────────────────────────────────────
    if function_name is None:
        if not parse.functions:
            print(f"Error: no functions found in {file_path}", file=sys.stderr)
            sys.exit(1)
        if len(parse.functions) > 1:
            print(
                f"Multiple functions found in {file_path}. "
                "Use --function to specify one:",
                file=sys.stderr,
            )
            for fn in parse.functions:
                print(f"  {fn}", file=sys.stderr)
            sys.exit(1)
        function_name = parse.functions[0]

    # ── Profile ──────────────────────────────────────────────────────────────
    profile = profile_file(file_path, function_name, timeout_s=timeout_s)

    # ── Fit ──────────────────────────────────────────────────────────────────
    fit = fit_curve(profile, parse)

    # ── Report ───────────────────────────────────────────────────────────────
    report = build_report(file_path, parse, profile, fit)

    # ── [Sprint 2 agent slot] ─────────────────────────────────────────────────
    # agent_summary = run_agent(report)  # ReAct loop over the three MCP tools

    # ── Display ──────────────────────────────────────────────────────────────
    print_report(report, function_name)


if __name__ == "__main__":
    main()
