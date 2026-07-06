"""config モジュールのユニットテスト。"""

import json
from pathlib import Path

import pytest

from config import VoiceNoteConfig, load_config, save_config


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
        assert config.save_folder == "/tmp/notes"
        assert config.transcription_mode == "local"

    def test_returns_none_for_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{ not valid json", encoding="utf-8")
        assert load_config(path) is None


class TestSaveConfig:
    def test_creates_parent_directories(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "config.json"
        save_config(path, VoiceNoteConfig(save_folder="/tmp/x"))
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["save_folder"] == "/tmp/x"

    def test_writes_unicode_without_escaping(self, tmp_path: Path):
        path = tmp_path / "config.json"
        save_config(path, VoiceNoteConfig(save_folder="/tmp/メモ"))
        raw = path.read_text(encoding="utf-8")
        assert "メモ" in raw

    def test_raises_runtime_error_on_failure(self, tmp_path: Path):
        # 書き込めない場所 (既存ファイルをディレクトリとして扱う)
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        with pytest.raises(RuntimeError):
            save_config(blocker / "child" / "config.json", VoiceNoteConfig())


class TestVoiceNoteConfigDefaults:
    def test_default_values(self):
        config = VoiceNoteConfig()
        assert config.save_folder == ""
        assert config.whisper_model == "small"
        assert config.transcription_mode == "local"
        assert config.vad_filter is True
        assert config.format_mode == "rule"
        assert config.openai_api_key is None


class TestToDict:
    def test_omits_none_api_key(self):
        config = VoiceNoteConfig(save_folder="/tmp")
        assert "openai_api_key" not in config.to_dict()

    def test_includes_api_key_when_set(self):
        config = VoiceNoteConfig(save_folder="/tmp", openai_api_key="sk-xxx")
        assert config.to_dict()["openai_api_key"] == "sk-xxx"

    def test_roundtrip(self):
        config = VoiceNoteConfig(
            save_folder="/tmp",
            whisper_model="large-v3",
            transcription_mode="openai",
            vad_filter=False,
            format_mode="llm",
            openai_api_key="sk-xxx",
        )
        assert VoiceNoteConfig.from_dict(config.to_dict()) == config


class TestFromDict:
    def test_legacy_vault_path_relative_save_folder_is_combined(self):
        config = VoiceNoteConfig.from_dict(
            {
                "vault_path": "/Users/x/Obsidian",
                "save_folder": "recordings",
            }
        )
        assert config.save_folder == "/Users/x/Obsidian/recordings"

    def test_legacy_vault_path_absolute_save_folder_kept_as_is(self):
        config = VoiceNoteConfig.from_dict(
            {
                "vault_path": "/Users/x/Obsidian",
                "save_folder": "/absolute/path",
            }
        )
        assert config.save_folder == "/absolute/path"

    def test_uses_default_transcription_mode_when_missing(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp"})
        assert config.transcription_mode == "local"

    def test_uses_default_vad_filter_when_missing(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp"})
        assert config.vad_filter is True

    def test_uses_default_format_mode_when_missing(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp"})
        assert config.format_mode == "rule"

    def test_preserves_existing_values(self):
        original = {
            "save_folder": "/tmp",
            "transcription_mode": "openai",
            "vad_filter": False,
            "format_mode": "llm",
        }
        config = VoiceNoteConfig.from_dict(original)
        assert config.transcription_mode == "openai"
        assert config.vad_filter is False
        assert config.format_mode == "llm"

    def test_does_not_mutate_input(self):
        original = {"save_folder": "/tmp"}
        snapshot = dict(original)
        VoiceNoteConfig.from_dict(original)
        assert original == snapshot

    def test_ignores_unknown_keys(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp", "unknown_key": "x"})
        assert config.save_folder == "/tmp"
