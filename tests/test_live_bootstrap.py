"""live.bootstrap(CLI と GUI で共有するライブ文字起こしの準備処理)のテスト。

エンジン生成・VAD・ローカルLLM はすべて注入した偽物で置き換える。
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from live import bootstrap
from live.bootstrap import (
    CPU_UNSUITABLE_MODELS,
    FALLBACK_ENGINE,
    VAD_FALLBACK_WARNING,
    BackendCheck,
    FallbackOnLoadEngine,
    LiveSetupError,
    build_engine,
    build_vad,
    check_backend,
    fallback_model,
)
from live.session_models import LiveSettings

MLX_REPO = "mlx-community/whisper-small-mlx"
HINT = "配布版のヒント"


class FakeUnavailable(RuntimeError):
    """EngineUnavailableError の代役"""


class EngineFactory:
    """create_engine の偽物。生成履歴を記録し、指定エンジンは失敗させる"""

    def __init__(self, errors: dict[str, Exception] | None = None):
        self.errors = dict(errors or {})
        self.calls: list[tuple[str, str]] = []
        self.notifies: list[object] = []

    def __call__(self, name, model, notify=None):
        self.calls = [*self.calls, (name, model)]
        self.notifies = [*self.notifies, notify]
        if name in self.errors:
            raise self.errors[name]
        return MagicMock(name=f"engine-{name}")


def _build(settings: LiveSettings, factory: EngineFactory, warnings: list[str], hint=HINT):
    return build_engine(
        settings,
        create_engine=factory,
        unavailable_error=FakeUnavailable,
        notify=lambda m: None,
        warn=warnings.append,
        install_hint=hint,
    )


class TestFallbackModel:
    @pytest.mark.parametrize("model", sorted(CPU_UNSUITABLE_MODELS))
    def test_cpu_unsuitable_models_become_default(self, model):
        assert fallback_model(model) == "small"

    def test_mlx_repo_becomes_default(self):
        assert fallback_model(MLX_REPO) == "small"

    @pytest.mark.parametrize("model", ["tiny", "medium", "large-v3"])
    def test_short_names_are_kept(self, model):
        assert fallback_model(model) == model


class TestBuildEngine:
    def test_faster_whisper_is_not_wrapped(self):
        factory = EngineFactory()
        settings = LiveSettings(engine="faster-whisper", model="large-v3-turbo")
        engine, used = _build(settings, factory, [])
        assert not isinstance(engine, FallbackOnLoadEngine)
        assert used is settings
        assert factory.calls == [("faster-whisper", "large-v3-turbo")]

    def test_notify_is_passed_to_create_engine(self):
        factory = EngineFactory()
        notify = MagicMock()
        build_engine(
            LiveSettings(engine="faster-whisper"),
            create_engine=factory,
            unavailable_error=FakeUnavailable,
            notify=notify,
            warn=print,
            install_hint=HINT,
        )
        assert factory.notifies == [notify]

    def test_mlx_is_wrapped_for_load_time_fallback(self):
        factory = EngineFactory()
        settings = LiveSettings(engine="mlx", model=MLX_REPO)
        warnings: list[str] = []
        engine, used = _build(settings, factory, warnings)
        assert isinstance(engine, FallbackOnLoadEngine)
        assert used is settings
        assert warnings == []

    def test_mlx_unavailable_falls_back_to_faster_whisper_small(self):
        factory = EngineFactory({"mlx": FakeUnavailable("mlx-whisper がありません")})
        settings = LiveSettings(engine="mlx", model=MLX_REPO)
        warnings: list[str] = []
        engine, used = _build(settings, factory, warnings)
        assert factory.calls == [("mlx", MLX_REPO), ("faster-whisper", "small")]
        assert (used.engine, used.model) == ("faster-whisper", "small")
        assert not isinstance(engine, FallbackOnLoadEngine)
        # 元の設定は変更しない(イミュータブル)
        assert (settings.engine, settings.model) == ("mlx", MLX_REPO)

    @pytest.mark.parametrize("model", ["large-v3-turbo", "turbo"])
    def test_turbo_falls_back_to_small(self, model):
        factory = EngineFactory({"mlx": FakeUnavailable("x")})
        _, used = _build(LiveSettings(engine="mlx", model=model), factory, [])
        assert used.model == "small"

    def test_creation_fallback_warning_contains_install_hint(self):
        factory = EngineFactory({"mlx": FakeUnavailable("mlx-whisper がありません")})
        warnings: list[str] = []
        _build(LiveSettings(engine="mlx"), factory, warnings, hint="特別な手順")
        assert len(warnings) == 1
        assert "特別な手順" in warnings[0]
        assert "mlx-whisper がありません" in warnings[0]
        assert "uv sync" not in warnings[0]

    def test_load_time_fallback_uses_install_hint_and_default_model(self):
        factory = EngineFactory()
        warnings: list[str] = []
        engine, _ = _build(LiveSettings(engine="mlx", model="large-v3-turbo"), factory, warnings)
        engine._primary.load.side_effect = FakeUnavailable("download failed")
        engine.load()
        assert factory.calls[-1] == ("faster-whisper", "small")
        assert any("download failed" in w and HINT in w for w in warnings)

    def test_explicit_faster_whisper_failure_raises_setup_error(self):
        cause = FakeUnavailable("壊れている")
        factory = EngineFactory({"faster-whisper": cause})
        with pytest.raises(LiveSetupError) as exc:
            _build(LiveSettings(engine="faster-whisper"), factory, [])
        assert str(exc.value) == "音声認識エンジンを利用できません: 壊れている"
        assert exc.value.__cause__ is cause

    def test_fallback_failure_raises_setup_error(self):
        factory = EngineFactory(
            {"mlx": FakeUnavailable("mlx なし"), "faster-whisper": FakeUnavailable("fw なし")}
        )
        warnings: list[str] = []
        with pytest.raises(LiveSetupError) as exc:
            _build(LiveSettings(engine="mlx"), factory, warnings)
        assert str(exc.value) == "音声認識エンジンを利用できません: fw なし"
        assert len(warnings) == 1

    def test_other_errors_propagate(self):
        factory = EngineFactory({"faster-whisper": OSError("disk")})
        with pytest.raises(OSError):
            _build(LiveSettings(engine="faster-whisper"), factory, [])


class TestFallbackOnLoadEngine:
    def test_default_install_hint(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        warnings: list[str] = []
        engine = FallbackOnLoadEngine(primary, MagicMock, FakeUnavailable, warnings.append)
        engine.load()
        assert bootstrap.DEFAULT_INSTALL_HINT in warnings[0]
        assert FALLBACK_ENGINE in warnings[0]

    def test_transcribe_delegates_to_active(self):
        primary = MagicMock()
        primary.transcribe.return_value = "p"
        engine = FallbackOnLoadEngine(primary, MagicMock, FakeUnavailable, print)
        engine.load()
        assert engine.transcribe("s", "ja") == "p"
        primary.transcribe.assert_called_once_with("s", "ja", prompt=None)

    def test_transcribe_forwards_prompt(self):
        primary = MagicMock()
        engine = FallbackOnLoadEngine(primary, MagicMock, FakeUnavailable, print)
        engine.load()
        engine.transcribe("s", "ja", prompt="参加者: 田中。")
        primary.transcribe.assert_called_once_with("s", "ja", prompt="参加者: 田中。")

    def test_transcribe_forwards_prompt_to_fallback(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        fallback = MagicMock()
        engine = FallbackOnLoadEngine(primary, lambda: fallback, FakeUnavailable, lambda _m: None)
        engine.load()
        engine.transcribe("s", None, prompt="用語: 予算。")
        fallback.transcribe.assert_called_once_with("s", None, prompt="用語: 予算。")

    def test_fallback_load_error_propagates(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        fallback = MagicMock()
        fallback.load.side_effect = FakeUnavailable("fallback も不可")
        engine = FallbackOnLoadEngine(primary, lambda: fallback, FakeUnavailable, lambda m: None)
        with pytest.raises(FakeUnavailable):
            engine.load()


class TestBuildVad:
    def _kind(self, kind):
        return lambda vad: kind

    def test_silero_fell_back_to_energy_warns(self):
        warnings: list[str] = []
        vad = MagicMock()
        result = build_vad(
            LiveSettings(vad="silero"),
            create_vad=lambda name: vad,
            vad_kind=self._kind("energy"),
            warn=warnings.append,
        )
        assert result is vad
        assert warnings == [VAD_FALLBACK_WARNING]

    def test_silero_success_has_no_warning(self):
        warnings: list[str] = []
        build_vad(
            LiveSettings(vad="silero"),
            create_vad=lambda name: MagicMock(),
            vad_kind=self._kind("silero"),
            warn=warnings.append,
        )
        assert warnings == []

    def test_explicit_energy_has_no_warning(self):
        warnings: list[str] = []
        requested: list[str] = []
        build_vad(
            LiveSettings(vad="energy"),
            create_vad=lambda name: requested.append(name) or MagicMock(),
            vad_kind=self._kind("energy"),
            warn=warnings.append,
        )
        assert warnings == []
        assert requested == ["energy"]


class FakeProbe:
    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message


class ClientFactory:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.created: list[tuple[MagicMock, tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        instance = MagicMock(name="client")
        self.created = [*self.created, (instance, args, kwargs)]
        return instance


def _check(settings, factory, probe_fn):
    return check_backend(settings, client_factory=factory, probe_fn=probe_fn, probe_timeout=1.5)


class TestCheckBackend:
    def test_disabled_skips_factory_and_probe(self):
        factory = ClientFactory()
        probe_fn = MagicMock()
        result = _check(LiveSettings(llm_enabled=False), factory, probe_fn)
        assert result == BackendCheck(backend=None, status="disabled", message="")
        assert factory.created == []
        probe_fn.assert_not_called()

    def test_invalid_url(self):
        probe_fn = MagicMock()
        result = _check(LiveSettings(), ClientFactory(ValueError("外部URLは不可")), probe_fn)
        assert result == BackendCheck(backend=None, status="invalid_url", message="外部URLは不可")
        probe_fn.assert_not_called()

    @pytest.mark.parametrize("status", ["server_down", "model_missing"])
    def test_probe_failure(self, status):
        factory = ClientFactory()
        result = _check(LiveSettings(), factory, lambda c, m: FakeProbe(status, "案内"))
        assert result == BackendCheck(backend=None, status=status, message="案内")
        assert len(factory.created) == 1

    def test_ok_returns_backend_with_default_timeout(self):
        factory = ClientFactory()
        probed: list[tuple] = []
        settings = LiveSettings(llm_base_url="http://127.0.0.1:11434", llm_model="m")

        def probe_fn(client, model):
            probed.append((client, model))
            return FakeProbe("ok", "接続しました")

        result = _check(settings, factory, probe_fn)
        (probe_client, p_args, p_kwargs), (backend, b_args, b_kwargs) = factory.created
        assert probed == [(probe_client, "m")]
        assert p_args == ("http://127.0.0.1:11434", "m")
        assert p_kwargs == {"timeout": 1.5}
        assert b_args == ("http://127.0.0.1:11434", "m")
        assert b_kwargs == {}
        assert result == BackendCheck(backend=backend, status="ok", message="接続しました")

    def test_backend_check_is_frozen(self):
        check = BackendCheck(backend=None, status="disabled", message="")
        with pytest.raises(FrozenInstanceError):
            check.status = "ok"  # type: ignore[misc]


# --- 実際に動いているエンジン名・解放 ---


class TestActiveEngineName:
    def test_primary_name_before_and_after_successful_load(self):
        primary = MagicMock()
        engine = FallbackOnLoadEngine(
            primary, MagicMock, FakeUnavailable, print, primary_name="mlx"
        )
        assert engine.active_engine_name == "mlx"
        engine.load()
        assert engine.active_engine_name == "mlx"

    def test_switches_to_fallback_name_after_load_failure(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        engine = FallbackOnLoadEngine(
            primary, MagicMock, FakeUnavailable, lambda _m: None, primary_name="mlx"
        )
        engine.load()
        assert engine.active_engine_name == FALLBACK_ENGINE

    def test_unload_delegates_to_active_engine(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        fallback = MagicMock()
        engine = FallbackOnLoadEngine(primary, lambda: fallback, FakeUnavailable, lambda _m: None)
        engine.load()
        engine.unload()
        fallback.unload.assert_called_once()
        primary.unload.assert_not_called()

    def test_unload_tolerates_engine_without_unload(self):
        engine = FallbackOnLoadEngine(object(), MagicMock, FakeUnavailable, print)
        engine.unload()

    def test_build_engine_sets_primary_name(self):
        engine, _ = _build(LiveSettings(engine="mlx", model=MLX_REPO), EngineFactory(), [])
        assert engine.active_engine_name == "mlx"


class TestEffectiveEngineName:
    def test_uses_active_engine_name(self):
        primary = MagicMock()
        primary.load.side_effect = FakeUnavailable("x")
        engine = FallbackOnLoadEngine(
            primary, MagicMock, FakeUnavailable, lambda _m: None, primary_name="mlx"
        )
        engine.load()
        settings = LiveSettings(engine="mlx")
        assert bootstrap.effective_engine_name(engine, settings) == FALLBACK_ENGINE

    def test_falls_back_to_settings_when_attribute_missing(self):
        settings = LiveSettings(engine="mlx")
        assert bootstrap.effective_engine_name(object(), settings) == "mlx"

    def test_ignores_non_string_attribute(self):
        settings = LiveSettings(engine="faster-whisper")
        assert bootstrap.effective_engine_name(MagicMock(), settings) == "faster-whisper"
