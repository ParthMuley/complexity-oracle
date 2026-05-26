"""Complexity Oracle — first-time setup wizard.

Handles API key configuration for the oracle CLI.

Usage:
  oracle setup               # interactive prompt, masked input
  oracle setup --key sk-...  # non-interactive (still confirms overwrite)
  oracle setup --key sk-... --force  # fully non-interactive (CI/scripts)
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".complexity_oracle"
ENV_FILE = CONFIG_DIR / ".env"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _key_exists() -> bool:
    """Return True if a key is already stored in the user-level env file."""
    if not ENV_FILE.exists():
        return False
    content = ENV_FILE.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("ANTHROPIC_API_KEY="):
            value = stripped.split("=", 1)[1].strip()
            return bool(value)
    return False


def _prompt_for_key() -> str:
    """Interactively prompt the user to paste their Anthropic API key (masked)."""
    try:
        key = getpass.getpass("Paste your Anthropic API key: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.", file=sys.stderr)
        sys.exit(0)
    if not key:
        print("Error: no key entered.", file=sys.stderr)
        sys.exit(1)
    return key


def _validate_key(key: str) -> tuple[bool, str]:
    """Make a real 1-token API call to verify the key works.

    Returns:
        (True, "") on success.
        (False, reason) on auth failure.

    Raises:
        RuntimeError on network / unexpected errors.
    """
    try:
        import anthropic  # local import — not needed at module load
    except ImportError:
        raise RuntimeError(
            "anthropic package is not installed. Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, ""
    except anthropic.AuthenticationError:
        return False, "API returned 401 — key is invalid or revoked."
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Anthropic API. Check your internet connection.\n({exc})"
        ) from exc
    except anthropic.APIStatusError as exc:
        # Treat any other 4xx/5xx as a validation failure with detail
        return False, f"API returned {exc.status_code}: {exc.message}"


def _save_key(key: str) -> None:
    """Write the API key to ~/.complexity_oracle/.env, creating the dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve any other lines in the file (e.g. future keys), only replace/add
    # the ANTHROPIC_API_KEY entry.
    existing_lines: list[str] = []
    if ENV_FILE.exists():
        existing_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    key_written = False
    for line in existing_lines:
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            new_lines.append(f"ANTHROPIC_API_KEY={key}")
            key_written = True
        else:
            new_lines.append(line)

    if not key_written:
        new_lines.append(f"ANTHROPIC_API_KEY={key}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _confirm_overwrite(force: bool) -> bool:
    """Ask the user whether to overwrite the existing key.  Returns True to proceed."""
    if force:
        return True
    try:
        answer = input(
            "A key is already configured. Overwrite? [y/N] "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.")
        sys.exit(0)
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_setup(key: str | None = None, force: bool = False) -> None:
    """Run the interactive (or scripted) setup wizard.

    Args:
        key:   If provided, skip the interactive prompt and use this value.
        force: If True, skip the overwrite confirmation prompt.
    """
    # ── Overwrite check ──────────────────────────────────────────────────────
    if _key_exists():
        if not _confirm_overwrite(force):
            print("Setup cancelled. Existing key kept.")
            return

    # ── Acquire key ──────────────────────────────────────────────────────────
    if key is None:
        key = _prompt_for_key()

    # ── Validate ─────────────────────────────────────────────────────────────
    print("Validating...", end=" ", flush=True)
    try:
        ok, reason = _validate_key(key)
    except RuntimeError as exc:
        print("✗")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not ok:
        print("✗")
        print(f"Error: {reason}", file=sys.stderr)
        print("Key not saved. Double-check and try again.", file=sys.stderr)
        sys.exit(1)

    print("✓")

    # ── Save ─────────────────────────────────────────────────────────────────
    _save_key(key)
    print(f"Saved to {ENV_FILE}")
    print("You're all set. Run:  oracle analyze myfile.py")
