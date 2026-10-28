"""config モジュールのユニットテスト。"""

import json
from pathlib import Path

import pytest

import config as config_module
from config import (
    DEFAULT_LIVE_LLM_BASE_URL,
    DEFAULT_LIVE_LLM_MODEL,
    DEFAULT_LIVE_MLX_WHISPER_MODEL,
    DEFAULT_LIVE_REFINE_MODEL,
    DEFAULT_LIVE_VAD,
    DEFAULT_LIVE_WHISPER_MODEL,
    InvalidConfigError,
    VoiceNoteConfig,
    configure_live_interactive,
    default_live_model_for,
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
        assert config.live_vad == "silero"
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
            live_refine_enabled=True,
            live_refine_model="large-v3-turbo",
        )
        save_config(path, original)
        assert load_config(path) == original

    def test_legacy_mlx_config_without_model_keeps_small(self, tmp_path: Path):
        """旧 config.json（mlx 指定・モデル未指定）は従来どおり small で読める。"""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"save_folder": "/tmp/x", "live_engine": "mlx"}), encoding="utf-8"
        )
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.live_engine == "mlx"
        assert loaded.live_whisper_model == "small"

    def test_legacy_config_without_live_vad_uses_silero(self, tmp_path: Path):
        """旧 config.json（live_vad キー無し）は新既定の silero で読み込む。"""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"save_folder": "/tmp/x", "live_engine": "faster-whisper"}),
            encoding="utf-8",
        )
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.live_vad == "silero" == DEFAULT_LIVE_VAD

    def test_explicit_energy_vad_is_respected(self, tmp_path: Path):
        """live_vad に energy が明示されていれば既定より優先する。"""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"save_folder": "/tmp/x", "live_vad": "energy"}), encoding="utf-8"
        )
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.live_vad == "energy"

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


class TestDefaultLiveModelFor:
    def test_mlx_defaults_to_large_v3_turbo(self):
        assert default_live_model_for("mlx") == "large-v3-turbo"
        assert DEFAULT_LIVE_MLX_WHISPER_MODEL == "large-v3-turbo"

    def test_faster_whisper_defaults_to_small(self):
        assert default_live_model_for("faster-whisper") == DEFAULT_LIVE_WHISPER_MODEL == "small"

    def test_unknown_engine_uses_global_default(self):
        assert default_live_model_for("bogus") == DEFAULT_LIVE_WHISPER_MODEL

    def test_global_defaults_are_unchanged(self):
        """エンジン別の既定は対話設定のみ。VoiceNoteConfig の既定は faster-whisper + small。"""
        config = VoiceNoteConfig()
        assert (config.live_engine, config.live_whisper_model) == ("faster-whisper", "small")


class TestLiveRefineConfig:
    def test_defaults(self):
        config = VoiceNoteConfig()
        assert config.live_refine_enabled is False
        assert config.live_refine_model == DEFAULT_LIVE_REFINE_MODEL == "large-v3"

    def test_legacy_config_without_refine_keys(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"save_folder": "/tmp/x", "live_engine": "mlx"}), encoding="utf-8"
        )
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.live_refine_enabled is False
        assert loaded.live_refine_model == "large-v3"

    def test_refine_keys_in_to_dict(self):
        data = VoiceNoteConfig(live_refine_enabled=True).to_dict()
        assert data["live_refine_enabled"] is True
        assert data["live_refine_model"] == "large-v3"


def _refine_prompts(monkeypatch, engine_choice: str, refine_answer: str | None):
    """エンジンだけ選び、再文字起こしの質問に refine_answer(None は既定値)で答えるスタブ"""
    refine_questions: list[tuple[str, dict]] = []
    fixed = iter(["y", engine_choice])

    def fake_ask(question, *args, **kwargs):
        if "高精度" in question:
            refine_questions.append((question, kwargs))
            return kwargs["default"] if refine_answer is None else refine_answer
        try:
            return next(fixed)
        except StopIteration:
            return kwargs.get("default", "")

    monkeypatch.setattr(config_module.Prompt, "ask", staticmethod(fake_ask))
    return refine_questions


class TestConfigureLiveRefine:
    def test_mlx_asks_and_defaults_to_no(self, monkeypatch):
        questions = _refine_prompts(monkeypatch, "1", None)
        overrides = configure_live_interactive()
        assert len(questions) == 1
        assert questions[0][1]["default"] == "n"
        assert overrides["live_refine_enabled"] is False

    def test_mlx_answer_yes(self, monkeypatch):
        _refine_prompts(monkeypatch, "1", "y")
        assert configure_live_interactive()["live_refine_enabled"] is True

    def test_faster_whisper_does_not_ask(self, monkeypatch):
        questions = _refine_prompts(monkeypatch, "2", "y")
        overrides = configure_live_interactive()
        assert questions == []
        assert overrides["live_refine_enabled"] is False


class TestConfigureLiveInteractive:
    def test_skip_returns_empty_overrides(self, monkeypatch):
        _fake_prompts(monkeypatch, ["n"])
        assert configure_live_interactive() == {}

    def test_collects_live_settings(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            [
                "y",
                "1",
                "large-v3-turbo",
                "energy",
                "ja",
                "qwen3:8b",
                "http://localhost:11434/v1",
                "y",
            ],
        )
        overrides = configure_live_interactive()
        assert overrides == {
            "live_engine": "mlx",
            "live_whisper_model": "large-v3-turbo",
            "live_vad": "energy",
            "live_language": "ja",
            "live_llm_model": "qwen3:8b",
            "live_llm_base_url": "http://localhost:11434/v1",
            "live_refine_enabled": True,
        }

    @pytest.mark.parametrize(
        "choice,expected_default", [("1", "large-v3-turbo"), ("2", "small")], ids=["mlx", "fw"]
    )
    def test_model_prompt_default_depends_on_engine(self, monkeypatch, choice, expected_default):
        defaults: list[str | None] = []

        answers = iter(
            [
                "y",
                choice,
                None,
                "silero",
                "",
                DEFAULT_LIVE_LLM_MODEL,
                DEFAULT_LIVE_LLM_BASE_URL,
                "n",
            ]
        )

        def fake_ask(question, *args, **kwargs):
            answer = next(answers)
            if answer is None:  # モデル入力: 既定値をそのまま受け入れる
                defaults.append(kwargs.get("default"))
                return kwargs["default"]
            return answer

        monkeypatch.setattr(config_module.Prompt, "ask", staticmethod(fake_ask))
        overrides = configure_live_interactive()
        assert defaults == [expected_default]
        assert overrides["live_whisper_model"] == expected_default

    def test_blank_model_falls_back_to_engine_default(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            ["y", "1", "  ", "silero", "", DEFAULT_LIVE_LLM_MODEL, DEFAULT_LIVE_LLM_BASE_URL, "y"],
        )
        assert configure_live_interactive()["live_whisper_model"] == "large-v3-turbo"

    def test_custom_model_is_kept(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            ["y", "2", " medium ", "silero", "", DEFAULT_LIVE_LLM_MODEL, DEFAULT_LIVE_LLM_BASE_URL],
        )
        assert configure_live_interactive()["live_whisper_model"] == "medium"

    def test_faster_whisper_choice_and_auto_language(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            [
                "y",
                "2",
                "small",
                "silero",
                "auto",
                DEFAULT_LIVE_LLM_MODEL,
                DEFAULT_LIVE_LLM_BASE_URL,
            ],
        )
        overrides = configure_live_interactive()
        assert overrides["live_engine"] == "faster-whisper"
        assert overrides["live_language"] == ""

    def test_invalid_engine_choice_is_reasked(self, monkeypatch):
        asked = _fake_prompts(
            monkeypatch,
            [
                "y",
                "9",
                "2",
                "small",
                "silero",
                "",
                DEFAULT_LIVE_LLM_MODEL,
                DEFAULT_LIVE_LLM_BASE_URL,
            ],
        )
        overrides = configure_live_interactive()
        assert overrides["live_engine"] == "faster-whisper"
        assert len(asked) == 8

    def test_non_loopback_base_url_is_reasked(self, monkeypatch):
        _fake_prompts(
            monkeypatch,
            [
                "y",
                "2",
                "small",
                "silero",
                "",
                "qwen2.5:7b",
                "http://example.com/v1",
                "http://127.0.0.1:11434/v1",
            ],
        )
        overrides = configure_live_interactive()
        assert overrides["live_llm_base_url"] == "http://127.0.0.1:11434/v1"


def _vad_prompts(monkeypatch, vad_answers: list[str | None]) -> tuple[list[str], list]:
    """VAD の質問だけ vad_answers を返し、他は既定値で答えるスタブ。

    None は既定値（kwargs["default"]）をそのまま受け入れる。
    Returns:
        (VAD 質問で渡された既定値の記録, VAD 質問の回数を数えるためのリスト)
    """
    vad_defaults: list = []
    remaining_vad = list(vad_answers)
    fixed = iter(["y", "2", "small"])

    def fake_ask(question, *args, **kwargs):
        if "VAD" in question:
            vad_defaults.append(kwargs.get("default"))
            answer = remaining_vad.pop(0)
            return kwargs["default"] if answer is None else answer
        try:
            return next(fixed)
        except StopIteration:
            return kwargs.get("default", "")

    monkeypatch.setattr(config_module.Prompt, "ask", staticmethod(fake_ask))
    return vad_defaults, remaining_vad


class TestConfigureLiveVad:
    def test_default_is_silero(self, monkeypatch):
        vad_defaults, _ = _vad_prompts(monkeypatch, [None])
        overrides = configure_live_interactive()
        assert vad_defaults == ["silero"]
        assert overrides["live_vad"] == "silero"

    @pytest.mark.parametrize("answer,expected", [("silero", "silero"), ("energy", "energy")])
    def test_each_choice(self, monkeypatch, answer, expected):
        _vad_prompts(monkeypatch, [answer])
        assert configure_live_interactive()["live_vad"] == expected

    def test_choice_is_normalized(self, monkeypatch):
        _vad_prompts(monkeypatch, ["  Energy "])
        assert configure_live_interactive()["live_vad"] == "energy"

    def test_invalid_choice_is_reasked(self, monkeypatch):
        vad_defaults, remaining = _vad_prompts(monkeypatch, ["webrtc", "vad", "energy"])
        overrides = configure_live_interactive()
        assert overrides["live_vad"] == "energy"
        assert len(vad_defaults) == 3
        assert remaining == []

    def test_vad_is_asked_after_engine_and_model(self, monkeypatch):
        asked = _fake_prompts(
            monkeypatch,
            ["y", "2", "small", "silero", "", DEFAULT_LIVE_LLM_MODEL, DEFAULT_LIVE_LLM_BASE_URL],
        )
        configure_live_interactive()
        assert "VAD" in asked[3]
