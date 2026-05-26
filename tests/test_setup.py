"""Tests for cli/setup.py — oracle setup wizard."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import complexity_oracle.cli.setup as setup_mod
from complexity_oracle.cli.setup import (
    _confirm_overwrite,
    _key_exists,
    _prompt_for_key,
    _save_key,
    _validate_key,
    run_setup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_file(tmp_path: Path, content: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


# ---------------------------------------------------------------------------
# _key_exists
# ---------------------------------------------------------------------------

class TestKeyExists:
    def test_returns_false_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_mod, "ENV_FILE", tmp_path / ".env")
        assert _key_exists() is False

    def test_returns_false_when_key_empty(self, tmp_path, monkeypatch):
        env_file = _make_env_file(tmp_path, "ANTHROPIC_API_KEY=\n")
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)
        assert _key_exists() is False

    def test_returns_true_when_key_present(self, tmp_path, monkeypatch):
        env_file = _make_env_file(tmp_path, "ANTHROPIC_API_KEY=sk-ant-abc123\n")
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)
        assert _key_exists() is True

    def test_returns_false_when_file_has_other_keys_only(self, tmp_path, monkeypatch):
        env_file = _make_env_file(tmp_path, "OTHER_KEY=foobar\n")
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)
        assert _key_exists() is False


# ---------------------------------------------------------------------------
# _save_key
# ---------------------------------------------------------------------------

class TestSaveKey:
    def test_creates_dir_and_file(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        env_file = config_dir / ".env"
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        _save_key("sk-ant-test")

        assert config_dir.exists()
        assert env_file.exists()

    def test_writes_correct_content(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        env_file = config_dir / ".env"
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        _save_key("sk-ant-test123")

        content = env_file.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-test123" in content

    def test_overwrites_existing_key_preserving_other_lines(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        config_dir.mkdir()
        env_file = _make_env_file(config_dir, "OTHER_VAR=hello\nANTHROPIC_API_KEY=old-key\n")
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        _save_key("sk-ant-new")

        content = env_file.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-new" in content
        assert "OTHER_VAR=hello" in content
        assert "old-key" not in content

    def test_appends_key_when_file_exists_without_it(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        config_dir.mkdir()
        env_file = _make_env_file(config_dir, "SOME_OTHER=value\n")
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        _save_key("sk-ant-appended")

        content = env_file.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-appended" in content
        assert "SOME_OTHER=value" in content


# ---------------------------------------------------------------------------
# _validate_key
# ---------------------------------------------------------------------------

class TestValidateKey:
    def test_returns_true_on_success(self):
        mock_client = MagicMock()
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            ok, reason = _validate_key("sk-ant-valid")

        assert ok is True
        assert reason == ""
        mock_client.messages.create.assert_called_once()

    def test_returns_false_on_auth_error(self):
        import anthropic as real_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = real_anthropic.AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401, headers={}),
            body={},
        )
        mock_anthropic = MagicMock(spec=real_anthropic)
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AuthenticationError = real_anthropic.AuthenticationError
        mock_anthropic.APIConnectionError = real_anthropic.APIConnectionError
        mock_anthropic.APIStatusError = real_anthropic.APIStatusError

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            ok, reason = _validate_key("sk-ant-bad")

        assert ok is False
        assert "401" in reason or "invalid" in reason.lower()

    def test_raises_on_connection_error(self):
        import anthropic as real_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = real_anthropic.APIConnectionError(
            request=MagicMock()
        )
        mock_anthropic = MagicMock(spec=real_anthropic)
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AuthenticationError = real_anthropic.AuthenticationError
        mock_anthropic.APIConnectionError = real_anthropic.APIConnectionError
        mock_anthropic.APIStatusError = real_anthropic.APIStatusError

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with pytest.raises(RuntimeError, match="Could not reach Anthropic API"):
                _validate_key("sk-ant-whatever")

    def test_raises_on_missing_anthropic_package(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
        with pytest.raises((RuntimeError, ImportError)):
            _validate_key("sk-ant-x")


# ---------------------------------------------------------------------------
# _confirm_overwrite
# ---------------------------------------------------------------------------

class TestConfirmOverwrite:
    def test_force_returns_true_without_prompt(self):
        with patch("builtins.input") as mock_input:
            result = _confirm_overwrite(force=True)
        assert result is True
        mock_input.assert_not_called()

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES"])
    def test_yes_answers_return_true(self, answer):
        with patch("builtins.input", return_value=answer):
            assert _confirm_overwrite(force=False) is True

    @pytest.mark.parametrize("answer", ["n", "N", "no", "", "  "])
    def test_no_or_empty_answers_return_false(self, answer):
        with patch("builtins.input", return_value=answer):
            assert _confirm_overwrite(force=False) is False


# ---------------------------------------------------------------------------
# _prompt_for_key
# ---------------------------------------------------------------------------

class TestPromptForKey:
    def test_returns_entered_key(self):
        with patch("getpass.getpass", return_value="sk-ant-entered"):
            result = _prompt_for_key()
        assert result == "sk-ant-entered"

    def test_strips_whitespace(self):
        with patch("getpass.getpass", return_value="  sk-ant-padded  "):
            result = _prompt_for_key()
        assert result == "sk-ant-padded"

    def test_exits_on_empty_input(self):
        with patch("getpass.getpass", return_value=""):
            with pytest.raises(SystemExit):
                _prompt_for_key()

    def test_exits_on_keyboard_interrupt(self):
        with patch("getpass.getpass", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit):
                _prompt_for_key()


# ---------------------------------------------------------------------------
# run_setup — end-to-end
# ---------------------------------------------------------------------------

class TestRunSetup:
    def _patch_env(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        env_file = config_dir / ".env"
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)
        return env_file

    def test_happy_path_saves_key(self, tmp_path, monkeypatch, capsys):
        env_file = self._patch_env(tmp_path, monkeypatch)

        with (
            patch("getpass.getpass", return_value="sk-ant-new"),
            patch.object(setup_mod, "_validate_key", return_value=(True, "")),
        ):
            run_setup()

        content = env_file.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-new" in content
        out = capsys.readouterr().out
        assert "Saved" in out
        assert "oracle analyze" in out

    def test_flag_skips_prompt(self, tmp_path, monkeypatch):
        self._patch_env(tmp_path, monkeypatch)

        with (
            patch("getpass.getpass") as mock_getpass,
            patch.object(setup_mod, "_validate_key", return_value=(True, "")),
        ):
            run_setup(key="sk-ant-flag")

        mock_getpass.assert_not_called()

    def test_invalid_key_exits_without_saving(self, tmp_path, monkeypatch):
        env_file = self._patch_env(tmp_path, monkeypatch)

        with (
            patch("getpass.getpass", return_value="sk-ant-bad"),
            patch.object(setup_mod, "_validate_key", return_value=(False, "API returned 401")),
            pytest.raises(SystemExit),
        ):
            run_setup()

        assert not env_file.exists()

    def test_overwrite_no_keeps_original(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        config_dir.mkdir()
        env_file = _make_env_file(config_dir, "ANTHROPIC_API_KEY=sk-ant-original\n")
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        with patch("builtins.input", return_value="n"):
            run_setup()

        content = env_file.read_text(encoding="utf-8")
        assert "sk-ant-original" in content

    def test_overwrite_yes_replaces_key(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        config_dir.mkdir()
        env_file = _make_env_file(config_dir, "ANTHROPIC_API_KEY=sk-ant-original\n")
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        with (
            patch("builtins.input", return_value="y"),
            patch("getpass.getpass", return_value="sk-ant-new"),
            patch.object(setup_mod, "_validate_key", return_value=(True, "")),
        ):
            run_setup()

        content = env_file.read_text(encoding="utf-8")
        assert "sk-ant-new" in content
        assert "sk-ant-original" not in content

    def test_force_flag_skips_overwrite_prompt(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".complexity_oracle"
        config_dir.mkdir()
        env_file = _make_env_file(config_dir, "ANTHROPIC_API_KEY=sk-ant-old\n")
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(setup_mod, "ENV_FILE", env_file)

        with (
            patch("builtins.input") as mock_input,
            patch("getpass.getpass", return_value="sk-ant-forced"),
            patch.object(setup_mod, "_validate_key", return_value=(True, "")),
        ):
            run_setup(force=True)

        mock_input.assert_not_called()
        content = env_file.read_text(encoding="utf-8")
        assert "sk-ant-forced" in content

    def test_network_error_exits_without_saving(self, tmp_path, monkeypatch):
        env_file = self._patch_env(tmp_path, monkeypatch)

        with (
            patch("getpass.getpass", return_value="sk-ant-x"),
            patch.object(
                setup_mod, "_validate_key",
                side_effect=RuntimeError("Could not reach Anthropic API"),
            ),
            pytest.raises(SystemExit),
        ):
            run_setup()

        assert not env_file.exists()
