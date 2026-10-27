"""live_cli.run_live と main_cli の --live 統合のテスト。

record_audio / LiveSession / create_engine / ローカルLLM はすべてモックする。
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

import live_cli
import main_cli
from config import VoiceNoteConfig
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

    def create_engine(name, model, notify=None):
        created_engines.append(name)
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
    session.stop.side_effect = lambda timeout=None: events.append("session.stop") or result
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

    def save_live_note(folder, body):
        events.append("save_live_note")
        note_bodies.append(body)
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
    e.engine_errors = engine_errors
    e.session = session
    e.session_cls = session_cls
    e.record_kwargs = record_kwargs
    e.note_bodies = note_bodies
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

    def test_prints_retranscribe_hint(self, env):
        live_cli.run_live(env.config, None, env.dest)
        text = env.out.getvalue()
        assert "--file" in text
        assert env.wav.name in text


class TestEngineFallback:
    def test_falls_back_to_faster_whisper(self, env):
        env.engine_errors["mlx"] = FakeUnavailable("mlx-whisper がありません")
        config = VoiceNoteConfig(save_folder=env.config.save_folder, live_engine="mlx")
        live_cli.run_live(config, None, env.dest)
        assert env.created_engines == ["mlx", "faster-whisper"]
        assert "uv sync --extra live-mlx" in env.out.getvalue()
        settings = env.session_cls.call_args.args[0]
        assert settings.engine == "faster-whisper"

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
