"""
live.engines(推論エンジン抽象)のテスト
faster_whisper / mlx_whisper はすべて偽装し、実機 mlx テストは環境変数で明示したときのみ実行する
"""

import os
import sys
import types
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import faster_whisper
import numpy as np
import pytest

from live import engines
from live.engines import (
    EngineUnavailableError,
    FasterWhisperEngine,
    MlxWhisperEngine,
    TimedText,
    TranscribedText,
    create_engine,
)


def _fw_segment(text, no_speech_prob=0.1, avg_logprob=-0.3):
    return types.SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


class _FakeWhisperModel:
    """faster_whisper.WhisperModel の偽物。生成と transcribe 呼び出しを記録する"""

    instances: list = []

    def __init__(self, model_name, device=None, compute_type=None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.calls: list = []
        self.segments = [_fw_segment("こんにちは")]
        self.language = "ja"
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        info = types.SimpleNamespace(language=self.language)
        return iter(self.segments), info


@pytest.fixture
def fake_fw(monkeypatch):
    _FakeWhisperModel.instances = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)
    return _FakeWhisperModel


@pytest.fixture
def samples():
    return np.linspace(-0.5, 0.5, 16000, dtype=np.float32)


# --- TranscribedText ---


class TestTranscribedText:
    def test_defaults_are_none(self):
        t = TranscribedText(text="abc")
        assert t.no_speech_prob is None
        assert t.avg_logprob is None
        assert t.language is None

    def test_is_frozen(self):
        t = TranscribedText(text="abc")
        with pytest.raises(FrozenInstanceError):
            t.text = "x"  # type: ignore[misc]


# --- FasterWhisperEngine ---


class TestFasterWhisperEngine:
    def test_load_creates_model_once_on_cpu_int8(self, fake_fw):
        engine = FasterWhisperEngine("small")
        engine.load()
        engine.load()
        assert len(fake_fw.instances) == 1
        model = fake_fw.instances[0]
        assert model.model_name == "small"
        assert model.device == "cpu"
        assert model.compute_type == "int8"

    def test_transcribe_without_load_loads_once(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, "ja")
        engine.transcribe(samples, "ja")
        assert len(fake_fw.instances) == 1
        assert len(fake_fw.instances[0].calls) == 2

    def test_passes_ndarray_directly(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, "ja")
        audio, _ = fake_fw.instances[0].calls[0]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        np.testing.assert_array_equal(audio, samples)

    def test_decode_options(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, "ja")
        _, kwargs = fake_fw.instances[0].calls[0]
        assert kwargs["language"] == "ja"
        assert 1 <= kwargs["beam_size"] <= 2
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["vad_filter"] is False

    def test_prompt_is_passed_as_initial_prompt(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, "ja", prompt="参加者: 田中。")
        _, kwargs = fake_fw.instances[0].calls[0]
        assert kwargs["initial_prompt"] == "参加者: 田中。"
        assert kwargs["condition_on_previous_text"] is False

    @pytest.mark.parametrize("prompt", [None, ""])
    def test_no_prompt_gives_none(self, fake_fw, samples, prompt):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, "ja", prompt=prompt)
        engine.transcribe(samples, "ja")
        for _, kwargs in fake_fw.instances[0].calls:
            assert kwargs["initial_prompt"] is None

    @pytest.mark.parametrize("language", [None, "en", "ja"])
    def test_language_is_passed_through(self, fake_fw, samples, language):
        engine = FasterWhisperEngine("small")
        engine.transcribe(samples, language)
        _, kwargs = fake_fw.instances[0].calls[0]
        assert kwargs["language"] == language

    def test_joins_multiple_segments(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.load()
        fake_fw.instances[0].segments = [
            _fw_segment("今日は", no_speech_prob=0.1, avg_logprob=-0.2),
            _fw_segment("晴れです。", no_speech_prob=0.4, avg_logprob=-0.6),
        ]
        result = engine.transcribe(samples, None)
        assert result.text == "今日は晴れです。"
        assert result.no_speech_prob == pytest.approx(0.4)
        assert result.avg_logprob == pytest.approx(-0.4)
        assert result.language == "ja"

    def test_strips_leading_space_of_text(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.load()
        fake_fw.instances[0].segments = [_fw_segment(" Hello"), _fw_segment(" world.")]
        result = engine.transcribe(samples, "en")
        assert result.text == "Hello world."

    def test_no_segments_returns_empty_text(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.load()
        fake_fw.instances[0].segments = []
        result = engine.transcribe(samples, "ja")
        assert result == TranscribedText(text="", language="ja")

    def test_empty_samples_skip_inference(self, fake_fw):
        engine = FasterWhisperEngine("small")
        result = engine.transcribe(np.zeros(0, dtype=np.float32), "ja")
        assert result.text == ""
        assert all(not m.calls for m in fake_fw.instances)

    def test_converts_non_float32_and_does_not_mutate_input(self, fake_fw):
        engine = FasterWhisperEngine("small")
        original = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        before = original.copy()
        engine.transcribe(original, "ja")
        audio, _ = fake_fw.instances[0].calls[0]
        assert audio.dtype == np.float32
        np.testing.assert_array_equal(original, before)

    def test_rejects_multi_dimensional_samples(self, fake_fw):
        engine = FasterWhisperEngine("small")
        with pytest.raises(ValueError):
            engine.transcribe(np.zeros((10, 2), dtype=np.float32), "ja")

    def test_notify_is_called_on_load(self, fake_fw):
        messages: list[str] = []
        engine = FasterWhisperEngine("small", notify=messages.append)
        engine.load()
        engine.load()
        assert len(messages) >= 1
        assert any("small" in m for m in messages)

    def test_load_failure_is_wrapped(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("download failed")

        monkeypatch.setattr(faster_whisper, "WhisperModel", boom)
        engine = FasterWhisperEngine("small")
        with pytest.raises(EngineUnavailableError, match="small"):
            engine.load()


# --- MlxWhisperEngine ---


@pytest.fixture
def fake_mlx(monkeypatch):
    """sys.modules に偽の mlx_whisper を差し込み、Apple Silicon 判定を真にする"""
    module = types.ModuleType("mlx_whisper")
    module.transcribe = MagicMock(
        return_value={
            "text": "こんにちは",
            "segments": [{"text": "こんにちは", "no_speech_prob": 0.2, "avg_logprob": -0.5}],
            "language": "ja",
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(engines, "_is_apple_silicon", lambda: True)
    return module


class TestMlxWhisperEngine:
    def test_module_import_is_lazy(self):
        # live.engines の import 時点では mlx_whisper を参照していない
        assert not hasattr(engines, "mlx_whisper")

    def test_load_warms_up_once(self, fake_mlx):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.load()
        engine.load()
        assert fake_mlx.transcribe.call_count == 1
        _, kwargs = fake_mlx.transcribe.call_args
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"

    def test_transcribe_call_arguments(self, fake_mlx, samples):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.transcribe(samples, "ja")
        args, kwargs = fake_mlx.transcribe.call_args_list[-1]
        audio = args[0]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        np.testing.assert_array_equal(audio, samples)
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
        assert kwargs["language"] == "ja"
        assert kwargs["condition_on_previous_text"] is False
        assert "beam_size" not in kwargs

    def test_prompt_is_passed_as_initial_prompt(self, fake_mlx, samples):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.transcribe(samples, "ja", prompt="用語: 予算。")
        _, kwargs = fake_mlx.transcribe.call_args_list[-1]
        assert kwargs["initial_prompt"] == "用語: 予算。"
        assert kwargs["condition_on_previous_text"] is False

    @pytest.mark.parametrize("prompt", [None, ""])
    def test_no_prompt_gives_none(self, fake_mlx, samples, prompt):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.transcribe(samples, "ja", prompt=prompt)
        _, kwargs = fake_mlx.transcribe.call_args_list[-1]
        assert kwargs["initial_prompt"] is None

    def test_warmup_has_no_prompt(self, fake_mlx):
        MlxWhisperEngine("mlx-community/whisper-small-mlx").load()
        _, kwargs = fake_mlx.transcribe.call_args
        assert kwargs.get("initial_prompt") is None

    def test_language_none_is_passed_through(self, fake_mlx, samples):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.transcribe(samples, None)
        _, kwargs = fake_mlx.transcribe.call_args_list[-1]
        assert kwargs["language"] is None

    def test_joins_multiple_segments(self, fake_mlx, samples):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.load()
        fake_mlx.transcribe.return_value = {
            "text": "今日は晴れです。",
            "segments": [
                {"text": "今日は", "no_speech_prob": 0.1, "avg_logprob": -0.2},
                {"text": "晴れです。", "no_speech_prob": 0.5, "avg_logprob": -0.4},
            ],
            "language": "ja",
        }
        result = engine.transcribe(samples, None)
        assert result.text == "今日は晴れです。"
        assert result.no_speech_prob == pytest.approx(0.5)
        assert result.avg_logprob == pytest.approx(-0.3)
        assert result.language == "ja"

    def test_no_segments_returns_empty_text(self, fake_mlx, samples):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.load()
        fake_mlx.transcribe.return_value = {"text": "", "segments": [], "language": "ja"}
        result = engine.transcribe(samples, "ja")
        assert result == TranscribedText(text="", language="ja")

    def test_short_model_name_maps_to_verified_repo(self, fake_mlx):
        engine = MlxWhisperEngine("large-v3-turbo")
        engine.load()
        _, kwargs = fake_mlx.transcribe.call_args
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"

    def test_unavailable_when_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlx_whisper", None)
        monkeypatch.setattr(engines, "_is_apple_silicon", lambda: True)
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        with pytest.raises(EngineUnavailableError, match="uv sync --extra live-mlx"):
            engine.load()

    def test_unavailable_on_non_apple_silicon(self, fake_mlx, monkeypatch):
        monkeypatch.setattr(engines, "_is_apple_silicon", lambda: False)
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        with pytest.raises(EngineUnavailableError, match="uv sync --extra live-mlx"):
            engine.load()
        fake_mlx.transcribe.assert_not_called()

    def test_load_failure_is_wrapped(self, fake_mlx):
        fake_mlx.transcribe.side_effect = OSError("repo not found")
        engine = MlxWhisperEngine("mlx-community/does-not-exist")
        with pytest.raises(EngineUnavailableError, match="does-not-exist"):
            engine.load()

    def test_empty_samples_skip_inference(self, fake_mlx):
        engine = MlxWhisperEngine("mlx-community/whisper-small-mlx")
        engine.load()
        result = engine.transcribe(np.zeros(0, dtype=np.float32), "ja")
        assert result.text == ""
        assert fake_mlx.transcribe.call_count == 1


class TestIsAppleSilicon:
    def test_true_on_darwin_arm64(self, monkeypatch):
        monkeypatch.setattr(engines.sys, "platform", "darwin")
        monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
        assert engines._is_apple_silicon() is True

    @pytest.mark.parametrize(("plat", "machine"), [("darwin", "x86_64"), ("linux", "arm64")])
    def test_false_otherwise(self, monkeypatch, plat, machine):
        monkeypatch.setattr(engines.sys, "platform", plat)
        monkeypatch.setattr(engines.platform, "machine", lambda: machine)
        assert engines._is_apple_silicon() is False


# --- create_engine ---


class TestCreateEngine:
    def test_faster_whisper(self, fake_fw):
        engine = create_engine("faster-whisper", "small")
        assert isinstance(engine, FasterWhisperEngine)

    def test_mlx(self, fake_mlx):
        engine = create_engine("mlx", "mlx-community/whisper-small-mlx")
        assert isinstance(engine, MlxWhisperEngine)

    def test_mlx_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(engines, "_is_apple_silicon", lambda: False)
        with pytest.raises(EngineUnavailableError):
            create_engine("mlx", "mlx-community/whisper-small-mlx")

    def test_mlx_not_installed_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlx_whisper", None)
        monkeypatch.setattr(engines, "_is_apple_silicon", lambda: True)
        with pytest.raises(EngineUnavailableError):
            create_engine("mlx", "mlx-community/whisper-small-mlx")

    @pytest.mark.parametrize("name", ["", "whisper", "MLX", "openai"])
    def test_unknown_name_raises_value_error(self, name):
        with pytest.raises(ValueError, match="エンジン"):
            create_engine(name, "small")

    def test_notify_is_forwarded(self, fake_fw):
        messages: list[str] = []
        engine = create_engine("faster-whisper", "small", notify=messages.append)
        engine.load()
        assert messages


# --- 実機 mlx(デフォルト skip) ---


@pytest.mark.mlx
@pytest.mark.skipif(
    os.environ.get("VOICENOTE_RUN_MLX") != "1",
    reason="実機 mlx テストは VOICENOTE_RUN_MLX=1 のときのみ実行",
)
def test_mlx_real_device_transcribes_silence():
    pytest.importorskip("mlx_whisper")
    repo = os.environ.get("VOICENOTE_MLX_MODEL", "mlx-community/whisper-small-mlx")
    engine = create_engine("mlx", repo)
    engine.load()
    result = engine.transcribe(np.zeros(16000, dtype=np.float32), "ja")
    assert isinstance(result, TranscribedText)
    assert isinstance(result.text, str)


# --- transcribe_timed / unload(停止後の高精度再文字起こし用) ---


def _timed_fw_segment(start, end, text, no_speech_prob=0.1, avg_logprob=-0.3):
    return types.SimpleNamespace(
        start=start, end=end, text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob
    )


class TestFasterWhisperTimed:
    def test_returns_timed_segments(self, fake_fw, samples):
        engine = FasterWhisperEngine("large-v3")
        engine.load()
        fake_fw.instances[0].segments = [
            _timed_fw_segment(0.0, 1.5, " こんにちは"),
            _timed_fw_segment(1.5, 3.0, "さようなら", no_speech_prob=0.7, avg_logprob=-1.2),
        ]
        result = engine.transcribe_timed(samples, "ja", prompt="用語: X。")
        assert result == (
            TimedText(0.0, 1.5, "こんにちは", 0.1, -0.3),
            TimedText(1.5, 3.0, "さようなら", 0.7, -1.2),
        )
        _, kwargs = fake_fw.instances[0].calls[-1]
        assert kwargs["initial_prompt"] == "用語: X。"
        assert kwargs["language"] == "ja"
        assert kwargs["condition_on_previous_text"] is False

    def test_empty_samples_return_empty(self, fake_fw):
        engine = FasterWhisperEngine("large-v3")
        assert engine.transcribe_timed(np.zeros(0, dtype=np.float32), None) == ()
        assert fake_fw.instances == []

    def test_unload_drops_model_and_reload_creates_new(self, fake_fw, samples):
        engine = FasterWhisperEngine("small")
        engine.load()
        engine.unload()
        assert engine._model is None
        engine.load()
        assert len(fake_fw.instances) == 2

    def test_unload_without_load_is_noop(self, fake_fw):
        FasterWhisperEngine("small").unload()


@pytest.fixture
def fake_mlx_runtime(monkeypatch, fake_mlx):
    """ModelHolder と mlx.core.clear_cache の偽物を sys.modules に差し込む"""
    holder = types.SimpleNamespace(model=object(), model_path="mlx-community/whisper-large-v3-mlx")
    transcribe_module = types.ModuleType("mlx_whisper.transcribe")
    transcribe_module.ModelHolder = holder
    core = types.ModuleType("mlx.core")
    core.clear_cache = MagicMock()
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", transcribe_module)
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return types.SimpleNamespace(holder=holder, core=core)


class TestMlxTimed:
    def test_returns_timed_segments(self, fake_mlx, samples):
        fake_mlx.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " 一つ目", "no_speech_prob": 0.1},
                {"start": 2.0, "end": 4.0, "text": "二つ目", "avg_logprob": -0.4},
            ],
            "language": "ja",
        }
        engine = MlxWhisperEngine("large-v3")
        result = engine.transcribe_timed(samples, "ja", prompt="参加者: 山田。")
        assert result == (
            TimedText(0.0, 2.0, "一つ目", 0.1, None),
            TimedText(2.0, 4.0, "二つ目", None, -0.4),
        )
        _, kwargs = fake_mlx.transcribe.call_args_list[-1]
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
        assert kwargs["initial_prompt"] == "参加者: 山田。"
        assert "beam_size" not in kwargs

    def test_empty_samples_return_empty(self, fake_mlx):
        engine = MlxWhisperEngine("large-v3")
        assert engine.transcribe_timed(np.zeros(0, dtype=np.float32), None) == ()

    def test_unload_clears_model_holder_and_cache(self, fake_mlx_runtime, fake_mlx):
        engine = MlxWhisperEngine("large-v3")
        engine.load()
        engine.unload()
        assert fake_mlx_runtime.holder.model is None
        assert fake_mlx_runtime.holder.model_path is None
        fake_mlx_runtime.core.clear_cache.assert_called_once()
        engine.load()  # 解放後の load はウォームアップをやり直す
        assert fake_mlx.transcribe.call_count == 2

    def test_unload_falls_back_to_metal_clear_cache(self, fake_mlx_runtime, fake_mlx):
        del fake_mlx_runtime.core.clear_cache
        metal = types.SimpleNamespace(clear_cache=MagicMock())
        fake_mlx_runtime.core.metal = metal
        MlxWhisperEngine("large-v3").unload()
        metal.clear_cache.assert_called_once()

    def test_unload_tolerates_missing_runtime(self, fake_mlx, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", None)
        monkeypatch.setitem(sys.modules, "mlx.core", None)
        MlxWhisperEngine("large-v3").unload()  # 例外にしない
