"""gui.live_settings_form（設定ダイアログのライブタブのロジック）のテスト。"""

from dataclasses import fields, replace

import pytest

from config import DEFAULT_LIVE_MLX_WHISPER_MODEL, DEFAULT_LIVE_WHISPER_MODEL, VoiceNoteConfig
from gui.live_settings_form import (
    LANGUAGE_AUTO,
    LOOPBACK_ONLY_MESSAGE,
    FormError,
    LiveFormValues,
    cpu_model_warning,
    engine_choices,
    field_label,
    form_values_from_config,
    model_after_engine_change,
    model_candidates,
    validate_live_form,
)
from live.engines import MLX_INSTALL_HINT, MLX_MODEL_REPOS

LIVE_KEYS = {f.name for f in fields(VoiceNoteConfig) if f.name.startswith("live_")}


def _values(**overrides) -> LiveFormValues:
    base = LiveFormValues(
        engine="faster-whisper",
        model="small",
        vad="silero",
        language=LANGUAGE_AUTO,
        llm_enabled=True,
        llm_model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        summary_interval="45",
    )
    return replace(base, **overrides)


def _base_config() -> VoiceNoteConfig:
    return VoiceNoteConfig(
        save_folder="/tmp/notes",
        recording_folder="/tmp/rec",
        whisper_model="medium",
        transcription_mode="openai",
        openai_api_key="sk-test",
        format_mode="none",
        vad_filter=False,
    )


# --- form_values_from_config ---


def test_form_values_from_default_config():
    values = form_values_from_config(VoiceNoteConfig())
    assert values == LiveFormValues(
        engine="faster-whisper",
        model="small",
        vad="silero",
        language=LANGUAGE_AUTO,
        llm_enabled=True,
        llm_model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        summary_interval="45",
    )


def test_form_values_keep_explicit_language_and_custom_values():
    config = VoiceNoteConfig(
        live_engine="mlx",
        live_whisper_model="large-v3-turbo",
        live_language="ja",
        live_vad="energy",
        live_llm_enabled=False,
        live_summary_interval_sec=90,
    )
    values = form_values_from_config(config)
    assert values.engine == "mlx"
    assert values.model == "large-v3-turbo"
    assert values.language == "ja"
    assert values.vad == "energy"
    assert values.llm_enabled is False
    assert values.summary_interval == "90"


def test_form_values_do_not_crash_on_hand_edited_config():
    config = VoiceNoteConfig(live_engine="whisper.cpp", live_language="  ", live_llm_enabled=1)
    values = form_values_from_config(config)
    assert values.engine == "whisper.cpp"
    assert values.language == LANGUAGE_AUTO
    assert values.llm_enabled is True


def test_form_values_round_trip_through_validation():
    config = replace(_base_config(), live_language="en", live_engine="mlx")
    overrides = validate_live_form(form_values_from_config(config), config)
    assert replace(config, **overrides) == config


def test_form_values_are_immutable():
    values = _values()
    with pytest.raises(AttributeError):
        values.engine = "mlx"  # type: ignore[misc]


# --- validate_live_form: 正常系と基本タブの保持 ---


def test_validate_returns_only_live_keys():
    overrides = validate_live_form(_values(), _base_config())
    assert isinstance(overrides, dict)
    assert set(overrides) == LIVE_KEYS


def test_validate_keeps_basic_tab_fields():
    base = _base_config()
    overrides = validate_live_form(_values(engine="mlx", model="medium"), base)
    saved = replace(base, **overrides)
    for name in (
        "save_folder",
        "recording_folder",
        "whisper_model",
        "transcription_mode",
        "openai_api_key",
        "format_mode",
        "vad_filter",
    ):
        assert getattr(saved, name) == getattr(base, name)
    assert saved.live_engine == "mlx"
    assert saved.live_whisper_model == "medium"


def test_validate_converts_values():
    overrides = validate_live_form(
        _values(model="  medium ", llm_enabled=False, summary_interval=" 30 ", vad="energy"),
        _base_config(),
    )
    assert overrides["live_whisper_model"] == "medium"
    assert overrides["live_llm_enabled"] is False
    assert overrides["live_summary_interval_sec"] == 30
    assert overrides["live_vad"] == "energy"


# --- base_url ---


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8080/v1"],
)
def test_validate_accepts_loopback_url(url):
    overrides = validate_live_form(_values(base_url=f"  {url} "), _base_config())
    assert overrides["live_llm_base_url"] == url


@pytest.mark.parametrize(
    "url", ["http://192.168.1.10:11434/v1", "https://api.openai.com/v1", "", "localhost:11434"]
)
def test_validate_rejects_non_loopback_url(url):
    result = validate_live_form(_values(base_url=url), _base_config())
    assert result == FormError("base_url", LOOPBACK_ONLY_MESSAGE)


def test_loopback_message_text():
    assert LOOPBACK_ONLY_MESSAGE == "ローカルのアドレス（localhost / 127.0.0.1）のみ指定できます"


def test_validate_rejects_non_loopback_url_even_if_llm_disabled():
    result = validate_live_form(
        _values(llm_enabled=False, base_url="http://example.com/v1"), _base_config()
    )
    assert isinstance(result, FormError)
    assert result.field == "base_url"


# --- 要約間隔 ---


@pytest.mark.parametrize("interval", ["0", "-1", "1.5", "abc", "", "4_5", "+45", "４５"])
def test_validate_rejects_invalid_interval(interval):
    result = validate_live_form(_values(summary_interval=interval), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "summary_interval"


def test_validate_accepts_positive_interval():
    overrides = validate_live_form(_values(summary_interval="1"), _base_config())
    assert overrides["live_summary_interval_sec"] == 1


# --- 言語 ---


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (LANGUAGE_AUTO, ""),
        ("", ""),
        ("auto", ""),
        ("AUTO", ""),
        ("ja", "ja"),
        ("en", "en"),
        (" EN ", "en"),
        ("yue", "yue"),
    ],
)
def test_validate_language(language, expected):
    overrides = validate_live_form(_values(language=language), _base_config())
    assert overrides["live_language"] == expected


@pytest.mark.parametrize("language", ["japanese", "j", "日本語", "ja-JP", "e n", "12"])
def test_validate_rejects_invalid_language(language):
    result = validate_live_form(_values(language=language), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "language"


# --- モデル・LLMモデル ---


def test_validate_rejects_empty_model():
    result = validate_live_form(_values(model="   "), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "model"


def test_validate_rejects_empty_llm_model_when_enabled():
    result = validate_live_form(_values(llm_model=" "), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "llm_model"


def test_validate_allows_empty_llm_model_when_disabled():
    overrides = validate_live_form(_values(llm_enabled=False, llm_model=""), _base_config())
    assert overrides["live_llm_model"] == ""


# --- 全体検証（LiveSettings.from_config） ---


def test_validate_rejects_unknown_engine():
    result = validate_live_form(_values(engine="whisper.cpp"), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "settings"
    assert "whisper.cpp" in result.message


def test_validate_rejects_unknown_vad():
    result = validate_live_form(_values(vad="webrtc"), _base_config())
    assert isinstance(result, FormError)
    assert result.field == "settings"
    assert "webrtc" in result.message


def test_validate_does_not_mutate_base_config():
    base = _base_config()
    validate_live_form(_values(engine="mlx"), base)
    assert base == _base_config()


# --- field_label ---


def test_field_label_known_and_unknown():
    assert field_label("base_url") == "LLMの接続先URL"
    assert field_label("summary_interval") == "要約間隔（秒）"
    assert field_label("settings") == "ライブ設定"
    assert field_label("unknown") == "unknown"


# --- model_after_engine_change ---


def test_engine_change_replaces_default_model_to_mlx_default():
    assert (
        model_after_engine_change("faster-whisper", "mlx", DEFAULT_LIVE_WHISPER_MODEL)
        == DEFAULT_LIVE_MLX_WHISPER_MODEL
    )


def test_engine_change_replaces_mlx_default_back_to_global_default():
    assert (
        model_after_engine_change("mlx", "faster-whisper", f" {DEFAULT_LIVE_MLX_WHISPER_MODEL} ")
        == DEFAULT_LIVE_WHISPER_MODEL
    )


@pytest.mark.parametrize(
    ("old", "new", "model"),
    [
        ("faster-whisper", "mlx", "medium"),
        ("mlx", "faster-whisper", "small"),
        ("mlx", "faster-whisper", "large-v3"),
    ],
)
def test_engine_change_keeps_user_edited_model(old, new, model):
    assert model_after_engine_change(old, new, model) == model


def test_engine_change_same_engine_keeps_model():
    assert model_after_engine_change("mlx", "mlx", DEFAULT_LIVE_MLX_WHISPER_MODEL) == (
        DEFAULT_LIVE_MLX_WHISPER_MODEL
    )


# --- cpu_model_warning ---


@pytest.mark.parametrize("model", ["large-v3-turbo", "turbo", " turbo "])
def test_cpu_warning_for_unsuitable_model(model):
    warning = cpu_model_warning("faster-whisper", model)
    assert warning is not None
    assert "small" in warning


@pytest.mark.parametrize(
    ("engine", "model"),
    [("faster-whisper", "small"), ("faster-whisper", "medium"), ("mlx", "large-v3-turbo")],
)
def test_no_cpu_warning(engine, model):
    assert cpu_model_warning(engine, model) is None


# --- engine_choices ---


def _spec_found(_name):
    return object()


def _spec_missing(_name):
    return None


def test_engine_choices_mlx_available():
    choices = engine_choices(find_spec=_spec_found, machine=lambda: "arm64", system="darwin")
    assert choices.values == ("mlx", "faster-whisper")
    assert choices.disabled == frozenset()
    assert choices.note is None
    assert choices.is_enabled("mlx")


def test_engine_choices_not_apple_silicon():
    choices = engine_choices(find_spec=_spec_found, machine=lambda: "x86_64", system="darwin")
    assert choices.disabled == frozenset({"mlx"})
    assert not choices.is_enabled("mlx")
    assert choices.is_enabled("faster-whisper")
    assert "Apple Silicon" in choices.note


def test_engine_choices_not_macos():
    choices = engine_choices(find_spec=_spec_found, machine=lambda: "arm64", system="linux")
    assert choices.disabled == frozenset({"mlx"})
    assert "Apple Silicon" in choices.note


def test_engine_choices_package_missing():
    requested = []

    def find_spec(name):
        requested.append(name)
        return None

    choices = engine_choices(find_spec=find_spec, machine=lambda: "arm64", system="darwin")
    assert requested == ["mlx_whisper"]
    assert choices.disabled == frozenset({"mlx"})
    assert MLX_INSTALL_HINT in choices.note


def test_engine_choices_find_spec_error_is_treated_as_missing():
    def find_spec(_name):
        raise ValueError("broken")

    choices = engine_choices(find_spec=find_spec, machine=lambda: "arm64", system="darwin")
    assert choices.disabled == frozenset({"mlx"})


def test_engine_choices_default_arguments_do_not_crash():
    choices = engine_choices()
    assert choices.values == ("mlx", "faster-whisper")


# --- model_candidates ---


def test_model_candidates_for_mlx_uses_short_name_map():
    assert model_candidates("mlx") == tuple(MLX_MODEL_REPOS)


def test_model_candidates_for_faster_whisper():
    candidates = model_candidates("faster-whisper")
    assert "small" in candidates
    assert "large-v3-turbo" not in candidates


def test_model_candidates_unknown_engine_falls_back_to_faster_whisper():
    assert model_candidates("unknown") == model_candidates("faster-whisper")


# --- 停止後の高精度再文字起こし ---


def test_form_values_refine_defaults():
    values = form_values_from_config(VoiceNoteConfig())
    assert values.refine_enabled is False
    assert values.refine_model == "large-v3"


def test_form_values_refine_from_config():
    config = VoiceNoteConfig(live_refine_enabled=True, live_refine_model="medium")
    values = form_values_from_config(config)
    assert values.refine_enabled is True
    assert values.refine_model == "medium"


def test_validate_refine_values():
    overrides = validate_live_form(
        _values(refine_enabled=True, refine_model=" large-v3 "), _base_config()
    )
    assert overrides["live_refine_enabled"] is True
    assert overrides["live_refine_model"] == "large-v3"


def test_validate_rejects_empty_refine_model():
    error = validate_live_form(_values(refine_model="  "), _base_config())
    assert isinstance(error, FormError)
    assert error.field == "refine_model"


def test_refine_field_labels():
    assert field_label("refine_enabled") != "refine_enabled"
    assert field_label("refine_model") != "refine_model"


def test_refine_model_candidates_are_mlx_models():
    from gui.live_settings_form import REFINE_ONLY_MLX_NOTE, refine_model_candidates

    assert refine_model_candidates() == tuple(MLX_MODEL_REPOS)
    assert "mlx" in REFINE_ONLY_MLX_NOTE
