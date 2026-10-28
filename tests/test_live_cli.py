"""live_cli.run_live と main_cli の --live 統合のテスト。

record_audio / LiveSession / create_engine / ローカルLLM はすべてモックする。
"""

import io
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

import live_cli
import main_cli
from config import VoiceNoteConfig
from live.refine import RefineResult
from live.session_models import LiveResult, LiveSegment
from live.view import LiveView


class FakeUnavailable(RuntimeError):
    """EngineUnavailableError の代役"""


class FakeProbe:
    def __init__(self, status: str, message: str = "msg"):
        self.status = status
        self.message = message


@pytest.fixture
def env(monkeypatch, tmp_path):
    """run_live の外部依存をすべて差し替え、呼び出し順序を events に記録する"""
    events: list[str] = []
    out = io.StringIO()
    monkeypatch.setattr(live_cli, "console", Console(file=out, width=400))

    wav = tmp_path / "2026-10-27_120000_recording.wav"
    created_engines: list[str] = []
    engine_errors: dict[str, Exception] = {}

    created_models: list[tuple[str, str]] = []

    def create_engine(name, model, notify=None):
        created_engines.append(name)
        created_models.append((name, model))
        if name in engine_errors:
            raise engine_errors[name]
        return MagicMock(name=f"engine-{name}")

    monkeypatch.setattr(live_cli, "_engine_api", lambda: (create_engine, FakeUnavailable))
    monkeypatch.setattr(live_cli, "create_vad", lambda name: MagicMock(name="vad"))

    result = LiveResult(
        segments=(LiveSegment(0, 0.5, 2.0, "テスト発話"),),
        final_summary="要約",
        dropped_chunks=0,
    )
    session = MagicMock(name="session")
    stop_kwargs: list[dict] = []

    def stop(timeout=None, **kwargs):
        events.append("session.stop")
        stop_kwargs.append(kwargs)
        return result

    session.stop.side_effect = stop
    session.wait_asr_idle.return_value = True
    session.start.side_effect = lambda: events.append("session.start")
    session_cls = MagicMock(return_value=session)
    monkeypatch.setattr(live_cli, "LiveSession", session_cls)

    record_kwargs: dict = {}

    def record_audio(**kwargs):
        record_kwargs.update(kwargs)
        events.append("record_audio")
        kwargs["on_start"]("Mic")
        kwargs["on_stop"]()
        kwargs["on_saved"](wav)
        events.append("on_saved")
        return wav

    monkeypatch.setattr(live_cli, "record_audio", record_audio)

    saved_note = tmp_path / "notes" / "x_live.md"
    note_bodies: list[str] = []

    note_audio_files: list[Path] = []

    def save_live_note(folder, body, audio_file=None):
        events.append("save_live_note")
        note_bodies.append(body)
        note_audio_files.append(audio_file)
        return saved_note

    monkeypatch.setattr(live_cli, "save_live_note", save_live_note)

    probe_status = {"value": "ok"}
    monkeypatch.setattr(live_cli, "LocalLLMClient", MagicMock(name="LocalLLMClient"))
    monkeypatch.setattr(
        live_cli, "probe", lambda backend, model: FakeProbe(probe_status["value"], "案内")
    )

    class Env:
        pass

    e = Env()
    e.events = events
    e.out = out
    e.wav = wav
    e.created_engines = created_engines
    e.created_models = created_models
    e.engine_errors = engine_errors
    e.session = session
    e.session_cls = session_cls
    e.stop_kwargs = stop_kwargs
    e.saved_note = saved_note
    e.result = result
    e.record_kwargs = record_kwargs
    e.note_bodies = note_bodies
    e.note_audio_files = note_audio_files
    e.probe_status = probe_status
    e.config = VoiceNoteConfig(save_folder=str(tmp_path / "notes"))
    e.dest = tmp_path / "Desktop"
    return e


class TestRunLiveOrder:
    def test_stop_order_invariant(self, env):
        live_cli.run_live(env.config, None, env.dest)
        assert env.events == [
            "session.start",
            "record_audio",
            "on_saved",
            "session.stop",
            "save_live_note",
        ]

    def test_record_audio_receives_session_feed_and_dest(self, env):
        live_cli.run_live(env.config, "2", env.dest)
        assert env.record_kwargs["chunk_sink"] is env.session.feed
        assert env.record_kwargs["dest_dir"] == env.dest
        assert env.record_kwargs["device"] == "2"

    def test_note_body_contains_result_and_wav(self, env):
        live_cli.run_live(env.config, None, env.dest)
        assert "テスト発話" in env.note_bodies[0]
        assert "要約" in env.note_bodies[0]
        assert str(env.wav) in env.note_bodies[0]

    def test_live_note_is_named_after_wav(self, env):
        live_cli.run_live(env.config, None, env.dest)
        assert env.note_audio_files == [env.wav]

    def test_prints_retranscribe_hint(self, env):
        live_cli.run_live(env.config, None, env.dest)
        text = env.out.getvalue()
        assert "--file" in text
        assert env.wav.name in text
        assert "OpenAI に送信します" not in text

    @pytest.mark.parametrize(
        "overrides",
        [{"transcription_mode": "openai"}, {"format_mode": "llm"}],
        ids=["openai", "llm"],
    )
    def test_hint_warns_external_send(self, env, overrides):
        config = VoiceNoteConfig(save_folder=env.config.save_folder, **overrides)
        live_cli.run_live(config, None, env.dest)
        text = env.out.getvalue()
        assert "OpenAI に送信します" in text
        assert "local / rule" in text

    def test_note_quality_warning_mentions_external_send(self, env):
        env.session.stop.side_effect = lambda timeout=None, **_kw: LiveResult(
            segments=(), final_summary="", dropped_chunks=2
        )
        config = VoiceNoteConfig(save_folder=env.config.save_folder, transcription_mode="openai")
        live_cli.run_live(config, None, env.dest)
        assert "OpenAI に送信します" in env.note_bodies[0]


class TestInterruptDuringStop:
    """record_audio が SIGINT ハンドラを戻した後の Ctrl+C(session.stop 中)"""

    def _interrupt(self, env):
        def stop(timeout=None, **_kw):
            env.events.append("session.stop")
            raise KeyboardInterrupt

        env.session.stop.side_effect = stop

    def test_interrupt_skips_note_and_keeps_wav(self, env):
        self._interrupt(env)
        live_cli.run_live(env.config, None, env.dest)  # 例外は外へ漏らさない
        text = env.out.getvalue()
        assert "ライブノート作成を中断しました（WAVは保存済み）" in text
        assert env.wav.name in text
        assert "--file" in text
        assert "save_live_note" not in env.events

    def test_interrupt_hint_warns_external_send(self, env):
        self._interrupt(env)
        config = VoiceNoteConfig(save_folder=env.config.save_folder, format_mode="llm")
        live_cli.run_live(config, None, env.dest)
        assert "OpenAI に送信します" in env.out.getvalue()


class TestEngineFallback:
    def test_falls_back_to_faster_whisper(self, env):
        env.engine_errors["mlx"] = FakeUnavailable("mlx-whisper がありません")
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_engine="mlx")
        live_cli.run_live(config, None, env.dest)
        assert env.created_engines == ["mlx", "faster-whisper"]
        assert "uv sync --extra live-mlx" in env.out.getvalue()
        settings = env.session_cls.call_args.args[0]
        assert settings.engine == "faster-whisper"

    MLX_REPO = "mlx-community/whisper-small-mlx"

    def test_fallback_does_not_pass_mlx_repo_to_faster_whisper(self, env):
        env.engine_errors["mlx"] = FakeUnavailable("mlx-whisper がありません")
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder, live_engine="mlx", live_whisper_model=self.MLX_REPO
        )
        live_cli.run_live(config, None, env.dest)
        assert env.created_models == [("mlx", self.MLX_REPO), ("faster-whisper", "small")]
        assert env.session_cls.call_args.args[0].model == "small"

    def test_fallback_keeps_short_model_name(self, env):
        env.engine_errors["mlx"] = FakeUnavailable("mlx-whisper がありません")
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder, live_engine="mlx", live_whisper_model="medium"
        )
        live_cli.run_live(config, None, env.dest)
        assert env.created_models[-1] == ("faster-whisper", "medium")

    @pytest.mark.parametrize("model", ["large-v3-turbo", "turbo"])
    def test_fallback_replaces_turbo_with_default_model(self, env, model):
        """faster-whisper(CPU) の turbo は実測 RTF 3〜4 で実時間に追いつかないため small に戻す"""
        env.engine_errors["mlx"] = FakeUnavailable("mlx-whisper がありません")
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder, live_engine="mlx", live_whisper_model=model
        )
        live_cli.run_live(config, None, env.dest)
        assert env.created_models == [("mlx", model), ("faster-whisper", "small")]
        assert env.session_cls.call_args.args[0].model == "small"

    def test_load_time_fallback_replaces_turbo_with_default_model(self, env):
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder,
            live_engine="mlx",
            live_whisper_model="large-v3-turbo",
        )
        live_cli.run_live(config, None, env.dest)
        engine = env.session_cls.call_args.kwargs["engine"]
        engine._primary.load.side_effect = FakeUnavailable("download failed")
        engine.load()
        assert env.created_models[-1] == ("faster-whisper", "small")

    def test_explicit_faster_whisper_turbo_is_respected(self, env):
        """faster-whisper を明示的に選んだ場合はフォールバックではないのでモデルを変えない"""
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder,
            live_engine="faster-whisper",
            live_whisper_model="large-v3-turbo",
        )
        live_cli.run_live(config, None, env.dest)
        assert env.created_models == [("faster-whisper", "large-v3-turbo")]

    def test_load_time_fallback_does_not_pass_mlx_repo(self, env):
        config = VoiceNoteConfig(
            save_folder=env.config.save_folder, live_engine="mlx", live_whisper_model=self.MLX_REPO
        )
        live_cli.run_live(config, None, env.dest)
        engine = env.session_cls.call_args.kwargs["engine"]
        engine._primary.load.side_effect = FakeUnavailable("download failed")
        engine.load()
        assert env.created_models[-1] == ("faster-whisper", "small")

    def test_exits_when_fallback_also_fails(self, env):
        env.engine_errors["faster-whisper"] = FakeUnavailable("壊れている")
        with pytest.raises(SystemExit) as exc:
            live_cli.run_live(env.config, None, env.dest)
        assert exc.value.code == 1
        assert "record_audio" not in env.events

    def test_invalid_settings_exit(self, env):
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_engine="bogus")
        with pytest.raises(SystemExit) as exc:
            live_cli.run_live(config, None, env.dest)
        assert exc.value.code == 1


VAD_FALLBACK_MESSAGE = "Silero VADを読み込めなかったため energy VAD にフォールバックしました"


class TestVadFallback:
    """Silero の読み込みに失敗して energy に落ちたことを画面に出す"""

    @pytest.fixture
    def real_vad(self, monkeypatch):
        """env が差し替えた create_vad を本物に戻す"""
        import live.vad as vad_module

        monkeypatch.setattr(live_cli, "create_vad", vad_module.create_vad)
        return vad_module

    def test_silero_load_failure_warns_and_uses_energy(self, env, real_vad, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("onnxruntime missing")

        monkeypatch.setattr(real_vad, "SileroVAD", broken)
        live_cli.run_live(env.config, None, env.dest)
        out = env.out.getvalue()
        assert VAD_FALLBACK_MESSAGE in out
        assert "入力が小さいと取りこぼしが増えます" in out
        assert isinstance(env.session_cls.call_args.kwargs["vad"], real_vad.EnergyVAD)
        assert "save_live_note" in env.events

    def test_warning_is_shown_before_recording(self, env, real_vad, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("onnxruntime missing")

        monkeypatch.setattr(real_vad, "SileroVAD", broken)
        live_cli.run_live(env.config, None, env.dest)
        out = env.out.getvalue()
        assert out.index(VAD_FALLBACK_MESSAGE) < out.index("ライブ文字起こしを開始します")

    def test_silero_success_has_no_warning(self, env, real_vad, monkeypatch):
        class FakeSilero(real_vad.SileroVAD):
            def __init__(self):
                pass

        monkeypatch.setattr(real_vad, "SileroVAD", FakeSilero)
        live_cli.run_live(env.config, None, env.dest)
        assert VAD_FALLBACK_MESSAGE not in env.out.getvalue()
        assert isinstance(env.session_cls.call_args.kwargs["vad"], FakeSilero)

    def test_explicit_energy_has_no_warning(self, env, real_vad):
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_vad="energy")
        live_cli.run_live(config, None, env.dest)
        assert VAD_FALLBACK_MESSAGE not in env.out.getvalue()
        assert isinstance(env.session_cls.call_args.kwargs["vad"], real_vad.EnergyVAD)

    def test_default_config_requests_silero(self, env, monkeypatch):
        requested: list[str] = []
        monkeypatch.setattr(
            live_cli, "create_vad", lambda name: requested.append(name) or MagicMock()
        )
        live_cli.run_live(env.config, None, env.dest)
        assert requested == ["silero"]


class TestLLMProbe:
    @pytest.mark.parametrize("status", ["server_down", "model_missing"])
    def test_unavailable_llm_continues_without_summary(self, env, status):
        env.probe_status["value"] = status
        live_cli.run_live(env.config, None, env.dest)
        assert env.session_cls.call_args.kwargs["backend"] is None
        assert "案内" in env.out.getvalue()
        assert "save_live_note" in env.events

    def test_ok_llm_passes_backend(self, env):
        live_cli.run_live(env.config, None, env.dest)
        assert env.session_cls.call_args.kwargs["backend"] is not None

    def test_probe_uses_short_timeout_but_summary_does_not(self, env, monkeypatch):
        """疎通確認だけ短いタイムアウトにし、録音開始を待たせない(要約は通常のタイムアウト)"""
        created: list[tuple[MagicMock, dict]] = []

        def client(base_url, model, **kwargs):
            instance = MagicMock(name="client")
            created.append((instance, kwargs))
            return instance

        probed: list[MagicMock] = []
        monkeypatch.setattr(live_cli, "LocalLLMClient", client)
        monkeypatch.setattr(
            live_cli, "probe", lambda backend, model: probed.append(backend) or FakeProbe("ok")
        )
        live_cli.run_live(env.config, None, env.dest)
        probe_client = probed[0]
        probe_kwargs = next(kw for inst, kw in created if inst is probe_client)
        assert probe_kwargs["timeout"] == live_cli.PROBE_TIMEOUT_SEC == 3.0
        backend = env.session_cls.call_args.kwargs["backend"]
        assert backend is not probe_client
        backend_kwargs = next(kw for inst, kw in created if inst is backend)
        assert "timeout" not in backend_kwargs

    def test_disabled_llm_skips_probe(self, env, monkeypatch):
        probe = MagicMock()
        monkeypatch.setattr(live_cli, "probe", probe)
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_llm_enabled=False)
        live_cli.run_live(config, None, env.dest)
        probe.assert_not_called()
        assert env.session_cls.call_args.kwargs["backend"] is None

    def test_non_loopback_url_continues_without_summary(self, env, monkeypatch):
        monkeypatch.setattr(live_cli, "LocalLLMClient", MagicMock(side_effect=ValueError("外部")))
        live_cli.run_live(env.config, None, env.dest)
        assert env.session_cls.call_args.kwargs["backend"] is None


class TestRecordingErrors:
    def test_record_error_stops_session_and_exits(self, env, monkeypatch):
        def failing(**kwargs):
            raise RuntimeError("録音データがありません")

        monkeypatch.setattr(live_cli, "record_audio", failing)
        with pytest.raises(SystemExit) as exc:
            live_cli.run_live(env.config, None, env.dest)
        assert exc.value.code == 1
        env.session.stop.assert_called_once()
        assert "save_live_note" not in env.events


class TestLiveView:
    def test_keeps_only_recent_segments(self):
        view = LiveView(max_segments=2)
        for i in range(3):
            view.add_segment(i, float(i), float(i) + 1, f"発話{i}")
        assert [s.text for s in view.recent_segments] == ["発話1", "発話2"]

    def test_render_includes_summary_and_status(self):
        view = LiveView(max_segments=5)
        view.add_segment(0, 61.0, 62.0, "こんにちは")
        view.set_summary("今の話題")
        view.set_status("録音中")
        out = io.StringIO()
        Console(file=out, width=100).print(view.render())
        text = out.getvalue()
        assert "こんにちは" in text
        assert "01:01" in text
        assert "今の話題" in text
        assert "録音中" in text


class TestMainCliLiveFlag:
    @pytest.mark.parametrize(
        "extra", [["--file", "a.wav"], ["--record-only"]], ids=["file", "record-only"]
    )
    def test_live_is_exclusive(self, monkeypatch, extra):
        load = MagicMock()
        monkeypatch.setattr(main_cli, "load_or_configure", load)
        monkeypatch.setattr(sys, "argv", ["main_cli.py", "--live", *extra])
        with pytest.raises(SystemExit) as exc:
            main_cli.main()
        assert exc.value.code == 1
        load.assert_not_called()

    def test_live_dispatches_to_run_live(self, monkeypatch):
        config = VoiceNoteConfig(save_folder="/tmp/x")
        monkeypatch.setattr(main_cli, "load_or_configure", lambda force_config: config)
        monkeypatch.setattr(main_cli, "setup_logging", lambda: None)
        run_live = MagicMock()
        monkeypatch.setattr(live_cli, "run_live", run_live)
        record = MagicMock()
        monkeypatch.setattr(main_cli, "record_audio", record)
        monkeypatch.setattr(sys, "argv", ["main_cli.py", "--live", "--device", "3"])
        main_cli.main()
        run_live.assert_called_once_with(config, "3", Path.home() / "Desktop")
        record.assert_not_called()


class TestLoadTimeFallback:
    """mlx はモデル取得失敗などで load() 時にも EngineUnavailableError を出しうる"""

    def _engines(self, primary_error: Exception | None):
        primary = MagicMock(name="primary")
        if primary_error is not None:
            primary.load.side_effect = primary_error
        primary.transcribe.return_value = "primary"
        fallback = MagicMock(name="fallback")
        fallback.transcribe.return_value = "fallback"
        return primary, fallback

    def test_uses_primary_when_load_succeeds(self):
        primary, fallback = self._engines(None)
        warnings: list[str] = []
        engine = live_cli.FallbackOnLoadEngine(
            primary, lambda: fallback, FakeUnavailable, warnings.append
        )
        engine.load()
        assert engine.transcribe("x", "ja") == "primary"
        fallback.load.assert_not_called()
        assert warnings == []

    def test_switches_to_fallback_on_unavailable(self):
        primary, fallback = self._engines(FakeUnavailable("download failed"))
        warnings: list[str] = []
        engine = live_cli.FallbackOnLoadEngine(
            primary, lambda: fallback, FakeUnavailable, warnings.append
        )
        engine.load()
        fallback.load.assert_called_once()
        assert engine.transcribe("x", None) == "fallback"
        assert any("download failed" in w for w in warnings)

    def test_other_errors_propagate(self):
        primary, fallback = self._engines(OSError("disk"))
        engine = live_cli.FallbackOnLoadEngine(primary, lambda: fallback, FakeUnavailable, print)
        with pytest.raises(OSError):
            engine.load()

    def test_run_live_wraps_non_default_engine(self, env):
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_engine="mlx")
        live_cli.run_live(config, None, env.dest)
        engine = env.session_cls.call_args.kwargs["engine"]
        assert isinstance(engine, live_cli.FallbackOnLoadEngine)

    def test_run_live_does_not_wrap_faster_whisper(self, env):
        live_cli.run_live(env.config, None, env.dest)
        engine = env.session_cls.call_args.kwargs["engine"]
        assert not isinstance(engine, live_cli.FallbackOnLoadEngine)


# ──────────────── 停止後の高精度再文字起こし ────────────────


class FakeModelFetch:
    """live_cli.model_fetch の代役(通信しない)"""

    def __init__(self, cached=True, fetch_error: Exception | None = None, events=None):
        self.cached = cached
        self.fetch_error = fetch_error
        self.fetched: list = []
        self.events = events if events is not None else []

    def resolve_repo(self, engine, model):
        from live.model_fetch import RepoSpec

        return RepoSpec(f"repo/{engine}/{model}")

    def is_cached(self, repo):
        return self.cached

    def fetch(self, repo, on_progress):
        self.events.append("fetch")
        if self.fetch_error is not None:
            raise self.fetch_error
        self.fetched.append(repo.repo_id)
        self.cached = True
        return "/cache"

    def describe_fetch_error(self, exc, required_bytes=None):
        return types.SimpleNamespace(message="取得できませんでした")


@pytest.fixture
def refine_env(env, monkeypatch):
    """mlx + 再文字起こし有効の設定と、refine_wav / overwrite / summarize_full の偽物"""
    fetch = FakeModelFetch(events=env.events)
    monkeypatch.setattr(live_cli, "model_fetch", fetch)
    fetch_threads: list = []

    class SyncThread:
        """確認・取得スレッドを start() で同期実行する(defer=True なら取得だけ走らせず取得中のまま)"""

        defer = False

        def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
            self.target, self.args, self.daemon, self.name = target, args, daemon, name
            self.alive = False
            fetch_threads.append(self)

        def start(self):
            if SyncThread.defer and self.name == "refine-model-fetch":
                self.alive = True
                return
            self.target(*self.args)

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(live_cli, "_fetch_thread_factory", SyncThread)
    refine_calls: list[dict] = []
    overwrites: list[tuple[Path, str, bool]] = []
    summaries: list[tuple[str, ...]] = []

    def refine_wav(wav, make_engine, release_live, **kwargs):
        env.events.append("refine_wav")
        refine_calls.append({"wav": wav, **kwargs})
        release_live()
        kwargs["on_progress"](0.0, 2.0, None)
        kwargs["on_progress"](2.0, 2.0, 0.0)
        segments = (LiveSegment(0, 0.0, 2.0, "高精度の発話"),)
        return RefineResult(segments, True, 1.0)

    def overwrite(path, body, participants=(), refined=False):
        env.events.append("overwrite_live_note")
        overwrites.append((path, body, refined))
        return path

    def summarize_full(backend, texts, context, **kwargs):
        summaries.append(tuple(texts))
        return "現在の話題: 高精度の要約"

    monkeypatch.setattr(live_cli, "refine_wav", refine_wav)
    monkeypatch.setattr(live_cli, "overwrite_live_note", overwrite)
    monkeypatch.setattr(live_cli, "summarize_full", summarize_full)
    env.session.wait_asr_idle.side_effect = lambda timeout: (
        env.events.append("wait_asr_idle") or True
    )
    env.fetch = fetch
    env.fetch_thread_cls = SyncThread
    env.fetch_threads = fetch_threads
    env.refine_calls = refine_calls
    env.overwrites = overwrites
    env.summaries = summaries
    env.config = VoiceNoteConfig(
        save_folder=env.config.save_folder,
        live_engine="mlx",
        live_whisper_model="large-v3-turbo",
        live_refine_enabled=True,
    )
    return env


class TestRefine:
    def test_order_fast_note_then_refine_then_overwrite(self, refine_env):
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert refine_env.events == [
            "session.start",
            "record_audio",
            "on_saved",
            "session.stop",
            "save_live_note",
            "wait_asr_idle",
            "refine_wav",
            "overwrite_live_note",
        ]
        # 速報版でも全文要約を省かない
        assert refine_env.stop_kwargs == [{}]

    def test_overwritten_note_has_both_transcripts_and_new_summary(self, refine_env):
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        path, body, refined = refine_env.overwrites[0]
        assert path == refine_env.saved_note
        assert refined is True
        assert "## 文字起こし（高精度: large-v3）" in body
        assert "高精度の発話" in body
        assert "## ライブ文字起こし（速報）" in body
        assert "テスト発話" in body
        assert "高精度の要約" in body
        assert refine_env.summaries == [("高精度の発話",)]
        assert "高精度版で上書きしました" in refine_env.out.getvalue()

    def test_refine_uses_settings_language_and_wav(self, refine_env):
        config = replace(refine_env.config, live_language="ja")
        live_cli.run_live(config, None, refine_env.dest)
        call = refine_env.refine_calls[0]
        assert call["wav"] == refine_env.wav
        assert call["language"] == "ja"
        assert call["hint"] is None

    def test_release_live_unloads_realtime_engine(self, refine_env):
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        engine = refine_env.session_cls.call_args.kwargs["engine"]
        assert isinstance(engine, live_cli.FallbackOnLoadEngine)
        engine._active.unload.assert_called_once()

    def test_progress_is_printed(self, refine_env):
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "100%" in refine_env.out.getvalue()

    def test_summary_failure_keeps_live_summary(self, refine_env, monkeypatch):
        monkeypatch.setattr(live_cli, "summarize_full", lambda *a, **k: None)
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        body = refine_env.overwrites[0][1]
        assert "要約" in body and "高精度の要約" not in body

    def test_disabled_does_not_refine(self, refine_env):
        config = replace(refine_env.config, live_refine_enabled=False)
        live_cli.run_live(config, None, refine_env.dest)
        assert "refine_wav" not in refine_env.events
        assert refine_env.stop_kwargs == [{}]

    def test_creation_fallback_skips_refine_with_file_hint(self, refine_env):
        refine_env.engine_errors["mlx"] = FakeUnavailable("no mlx")
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "refine_wav" not in refine_env.events
        assert refine_env.stop_kwargs == [{}]
        assert "--file" in refine_env.out.getvalue()

    def test_load_time_fallback_skips_refine(self, refine_env):
        def start():
            refine_env.events.append("session.start")
            engine = refine_env.session_cls.call_args.kwargs["engine"]
            engine._primary.load.side_effect = FakeUnavailable("download failed")
            engine.load()

        refine_env.session.start.side_effect = start
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "refine_wav" not in refine_env.events
        text = refine_env.out.getvalue()
        assert "faster-whisper" in text and "スキップ" in text

    def test_model_fetched_in_background_after_recording_starts(self, refine_env):
        refine_env.fetch.cached = False
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        events = refine_env.events
        assert events.index("record_audio") < events.index("fetch") < events.index("on_saved")
        assert refine_env.fetch.fetched == ["repo/mlx/large-v3"]
        names = [t.name for t in refine_env.fetch_threads]
        assert names == ["refine-model-check", "refine-model-fetch"]
        assert all(t.daemon is True for t in refine_env.fetch_threads)
        assert "refine_wav" in events

    def test_fetch_still_running_at_stop_skips_with_file_hint(self, refine_env):
        refine_env.fetch.cached = False
        refine_env.fetch_thread_cls.defer = True
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "refine_wav" not in refine_env.events
        text = refine_env.out.getvalue()
        assert "取得中" in text
        assert "--file" in text

    def test_fetch_failure_continues_without_refine(self, refine_env):
        refine_env.fetch.cached = False
        refine_env.fetch.fetch_error = OSError("secret /Users/x")
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "record_audio" in refine_env.events
        assert "refine_wav" not in refine_env.events
        text = refine_env.out.getvalue()
        assert "取得できませんでした" in text and "OSError" in text
        assert "secret" not in text

    def test_cached_model_starts_no_fetch_thread(self, refine_env):
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert [t.name for t in refine_env.fetch_threads] == ["refine-model-check"]

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("Could not determine home directory."), PermissionError("secret /Users/x")],
        ids=["runtime", "permission"],
    )
    def test_fetch_check_error_does_not_break_recording(self, refine_env, monkeypatch, error):
        def raising(engine, model):
            raise error

        monkeypatch.setattr(refine_env.fetch, "resolve_repo", raising)
        live_cli.run_live(refine_env.config, None, refine_env.dest)  # 例外を外へ漏らさない
        assert "save_live_note" in refine_env.events
        assert "refine_wav" not in refine_env.events
        text = refine_env.out.getvalue()
        assert f"確認できません（{type(error).__name__}）" in text
        assert "secret" not in text

    def test_check_thread_start_failure_does_not_break_recording(self, refine_env, monkeypatch):
        def no_threads(**_kwargs):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(live_cli, "_fetch_thread_factory", no_threads)
        refine_env.fetch.cached = False
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "save_live_note" in refine_env.events

    def test_ctrl_c_while_waiting_for_asr_keeps_fast_note(self, refine_env):
        def interrupted(timeout):
            raise KeyboardInterrupt

        refine_env.session.wait_asr_idle.side_effect = interrupted
        live_cli.run_live(refine_env.config, None, refine_env.dest)  # 例外を外へ漏らさない
        assert "save_live_note" in refine_env.events
        assert refine_env.overwrites == []
        assert "中断" in refine_env.out.getvalue()

    def test_ctrl_c_while_planning_keeps_fast_note(self, refine_env, monkeypatch):
        def interrupted(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(live_cli, "decide_refine", interrupted)
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "save_live_note" in refine_env.events
        assert "refine_wav" not in refine_env.events
        assert "中断" in refine_env.out.getvalue()

    def test_asr_not_idle_skips_refine(self, refine_env):
        refine_env.session.wait_asr_idle.side_effect = None
        refine_env.session.wait_asr_idle.return_value = False
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "refine_wav" not in refine_env.events
        assert "save_live_note" in refine_env.events

    def test_refine_error_keeps_fast_note(self, refine_env, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("secret transcript text")

        monkeypatch.setattr(live_cli, "refine_wav", broken)
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert refine_env.overwrites == []
        text = refine_env.out.getvalue()
        assert "RuntimeError" in text
        assert "secret transcript text" not in text

    def test_incomplete_refine_keeps_fast_note(self, refine_env, monkeypatch):
        monkeypatch.setattr(live_cli, "refine_wav", lambda *a, **k: RefineResult((), False, 1.0))
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert refine_env.overwrites == []

    def test_ctrl_c_cancels_and_keeps_fast_note(self, refine_env, monkeypatch):
        cancelled: list[bool] = []

        def refine_wav(wav, make_engine, release_live, **kwargs):
            refine_env.events.append("refine_wav")
            kwargs["cancel"].wait(timeout=5.0)
            cancelled.append(kwargs["cancel"].is_set())
            return RefineResult((), False, 1.0)

        joins = {"n": 0}
        real_join = live_cli._join_once

        def join_once(worker):
            joins["n"] += 1
            if joins["n"] == 1:
                raise KeyboardInterrupt
            real_join(worker)

        monkeypatch.setattr(live_cli, "refine_wav", refine_wav)
        monkeypatch.setattr(live_cli, "_join_once", join_once)
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert cancelled == [True]
        assert refine_env.overwrites == []
        assert "save_live_note" in refine_env.events
        assert "中断" in refine_env.out.getvalue()

    def test_overwrite_failure_reports_fast_note(self, refine_env, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("保存エラー: disk")

        monkeypatch.setattr(live_cli, "overwrite_live_note", broken)
        live_cli.run_live(refine_env.config, None, refine_env.dest)
        assert "速報版" in refine_env.out.getvalue()
