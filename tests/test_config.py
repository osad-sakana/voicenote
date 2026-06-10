"""config モジュールのユニットテスト。"""

import json
from pathlib import Path

import pytest

from config import _migrate, load_config, save_config


class TestLoadConfig:
    def test_returns_none_when_file_missing(self, tmp_path: Path):
        assert load_config(tmp_path / "nonexistent.json") is None

    def test_loads_existing_config(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "save_folder": "/tmp/notes",
                    "transcription_mode": "local",
                    "vad_filter": True,
                    "format_mode": "rule",
                }
            ),
            encoding="utf-8",
        )

        config = load_config(path)
        assert config is not None
        assert config["save_folder"] == "/tmp/notes"
        assert config["transcription_mode"] == "local"

    def test_returns_none_for_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{ not valid json", encoding="utf-8")
        assert load_config(path) is None


class TestSaveConfig:
    def test_creates_parent_directories(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "config.json"
        save_config(path, {"save_folder": "/tmp/x"})
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["save_folder"] == "/tmp/x"

    def test_writes_unicode_without_escaping(self, tmp_path: Path):
        path = tmp_path / "config.json"
        save_config(path, {"save_folder": "/tmp/メモ"})
        raw = path.read_text(encoding="utf-8")
        assert "メモ" in raw

    def test_raises_runtime_error_on_failure(self, tmp_path: Path):
        # 書き込めない場所 (既存ファイルをディレクトリとして扱う)
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        with pytest.raises(RuntimeError):
            save_config(blocker / "child" / "config.json", {})


class TestMigrate:
    def test_legacy_vault_path_relative_save_folder_is_combined(self):
        config = {
            "vault_path": "/Users/x/Obsidian",
            "save_folder": "recordings",
        }
        migrated = _migrate(config)
        assert "vault_path" not in migrated
        assert migrated["save_folder"] == "/Users/x/Obsidian/recordings"

    def test_legacy_vault_path_absolute_save_folder_kept_as_is(self):
        config = {
            "vault_path": "/Users/x/Obsidian",
            "save_folder": "/absolute/path",
        }
        migrated = _migrate(config)
        assert "vault_path" not in migrated
        assert migrated["save_folder"] == "/absolute/path"

    def test_adds_default_transcription_mode(self):
        migrated = _migrate({"save_folder": "/tmp"})
        assert migrated["transcription_mode"] == "local"

    def test_adds_default_vad_filter(self):
        migrated = _migrate({"save_folder": "/tmp"})
        assert migrated["vad_filter"] is True

    def test_adds_default_format_mode(self):
        migrated = _migrate({"save_folder": "/tmp"})
        assert migrated["format_mode"] == "rule"

    def test_preserves_existing_values(self):
        original = {
            "save_folder": "/tmp",
            "transcription_mode": "openai",
            "vad_filter": False,
            "format_mode": "llm",
        }
        migrated = _migrate(original)
        assert migrated["transcription_mode"] == "openai"
        assert migrated["vad_filter"] is False
        assert migrated["format_mode"] == "llm"

    def test_does_not_mutate_input(self):
        original = {"save_folder": "/tmp"}
        snapshot = dict(original)
        _migrate(original)
        assert original == snapshot
