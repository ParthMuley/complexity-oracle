from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Load .env files before any env-var reads.
# Precedence (first-one-wins with override=False):
#   1. local .env in cwd       — explicit project override
#   2. ~/.complexity_oracle/.env — saved by `oracle setup`
from dotenv import load_dotenv
load_dotenv()
load_dotenv(Path.home() / ".complexity_oracle" / ".env")

# Ensure stdout can handle Unicode (e.g. ═, ⚠, ✓) on Windows terminals.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from complexity_oracle.core.agent import run_agent
from complexity_oracle.core.fitter import fit_curve
from complexity_oracle.core.parser import parse_file
from complexity_oracle.core.profiler import profile_file
from complexity_oracle.core.report import build_report
from complexity_oracle.core.scanner import scan_folder
from complexity_oracle.cli.formatter import print_folder_report, print_report
from complexity_oracle.cli.setup import run_setup


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle",
        description="Detect hidden performance bugs by combining static and empirical analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a Python file for complexity.")
    analyze.add_argument("file", help="Path to a Python file or directory to analyze.")
    analyze.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=False,
        help="When analysing a folder, scan subdirectories recursively.",
    )
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
    analyze.add_argument(
        "--no-agent",
        action="store_true",
        default=False,
        help="Skip the AI agent step. Prints static + empirical results only (no LLM call).",
    )

    setup = sub.add_parser("setup", help="Configure your Anthropic API key.")
    setup.add_argument(
        "--key",
        default=None,
        metavar="SK_ANT_KEY",
        help="API key to save (skips the interactive prompt).",
    )
    setup.add_argument(
        "--force", "-f",
        action="store_true",
        default=False,
        help="Overwrite an existing key without asking for confirmation.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "setup":
        run_setup(key=args.key, force=args.force)
    elif args.command == "analyze":
        import os
        if os.path.isdir(args.file):
            _run_folder(args.file, args.recursive, args.timeout)
        else:
            _run_analyze(args.file, args.function, args.timeout, args.no_agent)


def _run_folder(
    folder_path: str,
    recursive: bool,
    timeout_s: float,
) -> None:
    import os
    if not os.path.isdir(folder_path):
        print(f"Error: not a directory — {folder_path}", file=sys.stderr)
        sys.exit(1)

    folder_report = scan_folder(folder_path, recursive=recursive, timeout_s=timeout_s)
    print_folder_report(folder_report)


def _run_analyze(
    file_path: str,
    function_name: str | None,
    timeout_s: float,
    no_agent: bool = False,
) -> None:
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

    # ── Agent (Sprint 2) ─────────────────────────────────────────────────────
    agent_result = None
    if not no_agent:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "Error: Anthropic API key not found.\n"
                "Run:  oracle setup          (to configure your key)\n"
                "Or:   oracle analyze --no-agent  (to skip the AI step)",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            agent_result = run_agent(file_path, function_name)
        except Exception as e:
            # Agent failure is non-fatal — show warning, display report without explanation
            print(f"Warning: agent failed ({e}). Showing static + empirical results only.", file=sys.stderr)

    # ── Display ──────────────────────────────────────────────────────────────
    print_report(report, function_name, agent_result)


if __name__ == "__main__":
    main()
