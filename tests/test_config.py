"""config モジュールのユニットテスト。"""

import json
from pathlib import Path

import pytest

import config as config_module
from config import (
    DEFAULT_LIVE_LLM_BASE_URL,
    DEFAULT_LIVE_LLM_MODEL,
    InvalidConfigError,
    VoiceNoteConfig,
    configure_live_interactive,
    load_config,
    resolve_api_key,
    save_config,
)


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

    def test_raises_invalid_config_error_for_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(InvalidConfigError):
            load_config(path)


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
        assert config.recording_folder == ""
        assert config.whisper_model == "small"
        assert config.transcription_mode == "local"
        assert config.vad_filter is True
        assert config.format_mode == "rule"
        assert config.openai_api_key is None
        assert config.live_engine == "faster-whisper"
        assert config.live_whisper_model == "small"
        assert config.live_language == ""
        assert config.live_vad == "energy"
        assert config.live_llm_enabled is True
        assert config.live_llm_base_url == "http://127.0.0.1:11434/v1"
        assert config.live_llm_model == "qwen2.5:7b"
        assert config.live_summary_interval_sec == 45

    def test_live_defaults_are_exposed_as_constants(self):
        config = VoiceNoteConfig()
        assert config.live_llm_base_url == DEFAULT_LIVE_LLM_BASE_URL
        assert config.live_llm_model == DEFAULT_LIVE_LLM_MODEL


class TestRecordingFolder:
    def test_round_trips_through_save_and_load(self, tmp_path: Path):
        path = tmp_path / "config.json"
        save_config(path, VoiceNoteConfig(recording_folder="/tmp/rec"))
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.recording_folder == "/tmp/rec"

    def test_missing_key_in_legacy_config_defaults_to_empty(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"save_folder": "/tmp/x"}), encoding="utf-8")
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.recording_folder == ""

    def test_included_in_to_dict(self):
        data = VoiceNoteConfig(recording_folder="/tmp/rec").to_dict()
        assert data["recording_folder"] == "/tmp/rec"


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

    def test_does_not_mutate_input_with_legacy_absolute_save_folder(self):
        original = {"vault_path": "/v", "save_folder": "/absolute"}
        snapshot = dict(original)
        VoiceNoteConfig.from_dict(original)
        assert original == snapshot

    def test_ignores_unknown_keys(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp", "unknown_key": "x"})
        assert config.save_folder == "/tmp"


class TestResolveApiKey:
    def test_prefers_env_over_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        config = VoiceNoteConfig(openai_api_key="sk-cfg")
        assert resolve_api_key(config) == "sk-env"

    def test_falls_back_to_config_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = VoiceNoteConfig(openai_api_key="sk-cfg")
        assert resolve_api_key(config) == "sk-cfg"

    def test_returns_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = VoiceNoteConfig()
        assert resolve_api_key(config) is None


class TestLiveConfigPersistence:
    def test_legacy_config_without_live_keys_uses_defaults(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"save_folder": "/tmp/x"}), encoding="utf-8")
        loaded = load_config(path)
        assert loaded is not None
        assert loaded == VoiceNoteConfig(save_folder="/tmp/x")

    def test_live_fields_round_trip(self, tmp_path: Path):
        path = tmp_path / "config.json"
        original = VoiceNoteConfig(
            save_folder="/tmp/x",
            live_engine="mlx",
            live_whisper_model="large-v3-turbo",
            live_language="ja",
            live_vad="silero",
            live_llm_enabled=False,
            live_llm_base_url="http://localhost:8080/v1",
            live_llm_model="qwen3:8b",
            live_summary_interval_sec=60,
        )
        save_config(path, original)
        assert load_config(path) == original

    def test_unknown_live_like_keys_are_ignored(self):
        config = VoiceNoteConfig.from_dict({"save_folder": "/tmp", "live_unknown": 1})
        assert config == VoiceNoteConfig(save_folder="/tmp")


def _fake_prompts(monkeypatch, answers: list[str]) -> list[str]:
    """Prompt.ask を順番に answers を返すスタブに差し替え、質問文の記録を返す"""
    asked: list[str] = []
    remaining = list(answers)

    def fake_ask(question, *args, **kwargs):
        asked.append(question)
        return remaining.pop(0)

    monkeypatch.setattr(config_module.Prompt, "ask", staticmethod(fake_ask))
    return asked


class TestConfigureLiveInteractive:
    def test_skip_returns_empty_overrides(self, monkeypatch):
        _fake_prompts(monkeypatch, ["n"])
        assert configure_live_interactive() == {}

    def test_collects_live_settings(self, monkeypatch):
        _fake_prompts(monkeypatch, ["y", "1", "ja", "qwen3:8b", "http://localhost:11434/v1"])
        overrides = configure_live_interactive()
        assert overrides == {
            "live_engine": "mlx",
            "live_language": "ja",
            "live_llm_model": "qwen3:8b",
            "live_llm_base_url": "http://localhost:11434/v1",
        }

    def test_faster_whisper_choice_and_auto_language(self, monkeypatch):
        _fake_prompts(
            monkeypatch, ["y", "2", "auto", DEFAULT_LIVE_LLM_MODEL, DEFAULT_LIVE_LLM_BASE_URL]
        )
        overrides = configure_live_interactive()
        assert overrides["live_engine"] == "faster-whisper"
        assert overrides["live_language"] == ""

    def test_invalid_engine_choice_is_reasked(self, monkeypatch):
        asked = _fake_prompts(
            monkeypatch, ["y", "9", "2", "", DEFAULT_LIVE_LLM_MODEL, DEFAULT_LIVE_LLM_BASE_URL]
        )
        overrides = configure_live_interactive()
        assert overrides["live_engine"] == "faster-whisper"
        assert len(asked) == 6

    def test_non_loopback_base_url_is_reasked(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            ["y", "2", "", "qwen2.5:7b", "http://example.com/v1", "http://127.0.0.1:11434/v1"],
        )
        overrides = configure_live_interactive()
        assert overrides["live_llm_base_url"] == "http://127.0.0.1:11434/v1"
