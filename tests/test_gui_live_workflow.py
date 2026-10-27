"""gui/live_workflow.py のテスト(実ウィンドウ・実音声・実ネットワークなし)

録音・セッション・エンジン・VAD・LLM・モデル取得はすべて Fake に差し替え、
スレッドは Immediate(同期実行)/ Deferred(捕捉のみ)で制御する。
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config import VoiceNoteConfig
from gui.live_state import LivePhase
from gui.live_workflow import (
    BANNER_ENGINE_FALLBACK,
    BANNER_EXTERNAL_SEND,
    BANNER_LOW_INPUT,
    BANNER_MODEL_DOWNLOAD,
    BANNER_OLLAMA,
    BANNER_VAD_FALLBACK,
    LiveDeps,
    LiveWorkflow,
    LiveWorkflowCallbacks,
    MeterReading,
    ModelFetcher,
    format_download_progress,
    validate_live_start,
)
from live.local_llm import ProbeResult
from live.model_fetch import RepoSpec
from live.session_models import LiveResult, LiveSegment

# ──────────────── スレッドのスタブ ────────────────


class ImmediateThread:
    """start() でターゲットを同期実行するスタブ"""

    def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
        self._target = target
        self._args = args
        self._done = False

    def start(self):
        try:
            self._target(*self._args)
        finally:
            self._done = True

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class DeferredThread:
    """start() では何もせず、run() で後からターゲットを実行できるスタブ"""

    instances: list[DeferredThread] = []

    def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
        self.target = target
        self.args = args
        self.started = False
        self.finished = False
        DeferredThread.instances.append(self)

    def start(self):
        self.started = True

    def run(self):
        try:
            self.target(*self.args)
        finally:
            self.finished = True

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self.started and not self.finished


@pytest.fixture(autouse=True)
def _reset_deferred():
    DeferredThread.instances = []
    yield
    DeferredThread.instances = []


def run_pending() -> None:
    """捕捉済みで未実行の Deferred スレッドを順に実行する(実行中に増えた分も含む)"""
    while True:
        pending = [t for t in DeferredThread.instances if t.started and not t.finished]
        if not pending:
            return
        pending[0].run()


# ──────────────── Fake 部品 ────────────────


class Unavailable(Exception):
    pass


class FakeEngine:
    def __init__(self, name, model, fail_load=False):
        self.name = name
        self.model = model
        self.fail_load = fail_load
        self.loaded = False
        self.load_gate: threading.Event | None = None  # 設定すると load() がこれを待つ
        self.load_entered = threading.Event()

    def load(self):
        self.load_entered.set()
        if self.load_gate is not None:
            self.load_gate.wait(timeout=5.0)
        if self.fail_load:
            raise Unavailable(f"{self.name} を読み込めません")
        self.loaded = True

    def transcribe(self, samples, language):
        return f"{self.name}:{language}"


class FakeEngineApi:
    def __init__(self, unavailable: set[str] | None = None, load_fails: set[str] | None = None):
        self.unavailable = unavailable or set()
        self.load_fails = load_fails or set()
        self.created: list[FakeEngine] = []

    def create_engine(self, name, model, notify=None):
        if name in self.unavailable:
            raise Unavailable(f"{name} は使えません")
        engine = FakeEngine(name, model, fail_load=name in self.load_fails)
        self.created.append(engine)
        return engine


class FakeVad:
    def __init__(self, kind):
        self.kind = kind


def fake_create_vad_factory(fallback: bool):
    def create_vad(name):
        if name == "silero" and fallback:
            return FakeVad("energy")
        return FakeVad(name)

    return create_vad


def fake_vad_kind(vad):
    return vad.kind


class FakeClient:
    def __init__(self, base_url, model, timeout=None):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout


def probe_returning(status: str, message: str):
    def _probe(client, model):
        return ProbeResult(status=status, message=message)

    return _probe


class FakeFetcher:
    def __init__(self, cached=True, fail: Exception | None = None, progress=()):
        self.cached = cached
        self.fail = fail
        self.progress = progress
        self.fetched: list[RepoSpec] = []
        self.resolved: list[tuple[str, str]] = []

    def resolve_repo(self, engine, model):
        self.resolved.append((engine, model))
        if model == "unknown-model":
            raise ValueError(f"faster-whisper の不明なモデル名です: {model}")
        return RepoSpec(f"repo/{model}")

    def is_cached(self, repo):
        return self.cached

    def fetch(self, repo, on_progress):
        self.fetched.append(repo)
        for done, total in self.progress:
            on_progress(done, total)
        if self.fail is not None:
            raise self.fail
        return "/cache/path"

    def build(self) -> ModelFetcher:
        return ModelFetcher(
            resolve_repo=self.resolve_repo, is_cached=self.is_cached, fetch=self.fetch
        )


class FakeRecorder:
    def __init__(
        self,
        device_id,
        dest,
        chunk_sink,
        log,
        finalized_path: Path | None,
        fail_start=False,
        stop_result=True,
    ):
        self.device_id = device_id
        self.dest = dest
        self.chunk_sink = chunk_sink
        self.log = log
        self.finalized_path = finalized_path
        self.fail_start = fail_start
        self.stop_result = stop_result
        self.started = False
        self.finalize_timeouts: list[float] = []
        self.stop_called = False
        self.closing = False
        self.on_stop = None  # stop_with_timeout() の途中で呼ぶフック(ハング中の操作を模す)

    def start(self):
        self.log.append("recorder.start")
        if self.fail_start:
            raise RuntimeError("デバイスが使用できません")
        self.started = True

    def finalize_recording(self, timeout=2.0):
        self.log.append("finalize")
        self.finalize_timeouts.append(timeout)
        return self.finalized_path

    def stop_with_timeout(self, timeout=5.0):
        self.log.append("stop_with_timeout")
        self.stop_called = True
        if self.on_stop is not None:
            self.on_stop()
        return self.stop_result

    def is_closing(self):
        return self.closing


class FakeSession:
    def __init__(
        self,
        settings,
        engine,
        vad,
        backend,
        callbacks,
        log,
        result: LiveResult | None = None,
        fail_stop=False,
    ):
        self.settings = settings
        self.engine = engine
        self.vad = vad
        self.backend = backend
        self.callbacks = callbacks
        self.log = log
        self.result = result or LiveResult(
            segments=(LiveSegment(0, 0.0, 1.0, "こんにちは"),),
            final_summary="要約",
            dropped_chunks=0,
        )
        self.fail_stop = fail_stop
        self.fed: list[np.ndarray] = []
        self.started = False
        self.stop_timeouts: list[float] = []
        self.backlog_sec = 1.5
        self.dropped_chunks = 3

    def start(self):
        self.log.append("session.start")
        self.started = True

    def feed(self, chunk):
        self.fed.append(chunk)

    def stop(self, timeout=30.0):
        self.log.append("session.stop")
        self.stop_timeouts.append(timeout)
        if self.fail_stop:
            raise RuntimeError("stop 失敗")
        return self.result


class SpyCallbacks:
    def __init__(self, log):
        self.log = log
        self.states = []
        self.logs: list[str] = []
        self.recording_started = 0
        self.wav_saved: list[Path] = []
        self.done: list[tuple[Path, Path]] = []
        self.errors: list[str] = []

    def _started(self):
        self.recording_started += 1

    def _wav_saved(self, path):
        self.log.append("on_wav_saved")
        self.wav_saved.append(path)

    def _done(self, wav, note):
        self.log.append("on_done")
        self.done.append((wav, note))

    def build(self) -> LiveWorkflowCallbacks:
        return LiveWorkflowCallbacks(
            on_state=self.states.append,
            on_log=self.logs.append,
            on_recording_started=self._started,
            on_wav_saved=self._wav_saved,
            on_done=self._done,
            on_error=self.errors.append,
        )


class Harness:
    """LiveWorkflow と Fake 群をまとめて組み立てる"""

    def __init__(
        self,
        tmp_path: Path,
        *,
        config: VoiceNoteConfig | None = None,
        unavailable: set[str] | None = None,
        load_fails: set[str] | None = None,
        vad_fallback=False,
        probe=None,
        fetcher: FakeFetcher | None = None,
        finalized: Path | None | str = "default",
        fail_start=False,
        stop_result=True,
        session_result: LiveResult | None = None,
        session_fail_stop=False,
        session_factory_error: Exception | None = None,
        can_start=lambda: True,
        thread_factory=ImmediateThread,
        note_saver=None,
    ):
        self.tmp_path = tmp_path
        self.log: list[str] = []
        self.config = config or VoiceNoteConfig(
            save_folder=str(tmp_path / "notes"), live_engine="faster-whisper", live_vad="silero"
        )
        self.engine_api = FakeEngineApi(unavailable, load_fails)
        self.fetcher = fetcher or FakeFetcher()
        self.wav = (
            tmp_path / "2026-10-27_120000_recording.wav" if finalized == "default" else finalized
        )
        self.fail_start = fail_start
        self.stop_result = stop_result
        self.session_result = session_result
        self.session_fail_stop = session_fail_stop
        self.session_factory_error = session_factory_error
        self.recorders: list[FakeRecorder] = []
        self.sessions: list[FakeSession] = []
        self.saved_notes: list[tuple[Path, str, Path | None]] = []
        self.spy = SpyCallbacks(self.log)
        deps = LiveDeps(
            recorder_factory=self._make_recorder,
            session_factory=self._make_session,
            create_engine=self.engine_api.create_engine,
            unavailable_error=Unavailable,
            create_vad=fake_create_vad_factory(vad_fallback),
            vad_kind=fake_vad_kind,
            client_factory=FakeClient,
            probe_fn=probe or probe_returning("ok", "ローカルLLM（qwen）に接続しました。"),
            model_fetcher=self.fetcher.build(),
            note_saver=note_saver or self._save_note,
        )
        self.clock_now = 100.0
        self.wf = LiveWorkflow(
            self.config,
            self.spy.build(),
            deps=deps,
            can_start=can_start,
            thread_factory=thread_factory,
            clock=lambda: self.clock_now,
        )

    def _make_recorder(self, device_id, dest, chunk_sink):
        recorder = FakeRecorder(
            device_id,
            dest,
            chunk_sink,
            self.log,
            self.wav,
            fail_start=self.fail_start,
            stop_result=self.stop_result,
        )
        self.recorders.append(recorder)
        return recorder

    def _make_session(self, settings, *, engine, vad, backend, callbacks):
        if self.session_factory_error is not None:
            raise self.session_factory_error
        session = FakeSession(
            settings,
            engine,
            vad,
            backend,
            callbacks,
            self.log,
            result=self.session_result,
            fail_stop=self.session_fail_stop,
        )
        self.sessions.append(session)
        return session

    def _save_note(self, folder, body, audio_file=None):
        self.log.append("save")
        note = Path(folder) / f"note{len(self.saved_notes)}_live.md"
        self.saved_notes.append((Path(folder), body, audio_file))
        return note

    def start(self, device_id=1, label="[1] マイク"):
        return self.wf.start(device_id, label, self.tmp_path)

    @property
    def state(self):
        return self.wf.state

    def banner(self, kind):
        return next((b for b in self.state.banners if b.kind == kind), None)


# ──────────────── validate_live_start ────────────────


class TestValidateLiveStart:
    def test_ok(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        assert validate_live_start(config, "[1] マイク", mic_busy=False) is None

    def test_requires_save_folder(self):
        error = validate_live_start(VoiceNoteConfig(), "[1] マイク", mic_busy=False)
        assert error is not None
        assert error[0] == "設定が必要"

    def test_rejects_missing_device(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "デバイスなし", mic_busy=False)
        assert error is not None
        assert "デバイス" in error[0]

    def test_rejects_when_mic_busy(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(config, "[1] マイク", mic_busy=True)
        assert error is not None
        assert "実行中" in error[1]

    def test_rejects_invalid_live_settings(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_engine="bogus")
        error = validate_live_start(config, "[1] マイク", mic_busy=False)
        assert error is not None
        assert "bogus" in error[1]


class TestFormatDownloadProgress:
    def test_unknown_total(self):
        assert "small" in format_download_progress("small", 0, 0)

    def test_with_total_shows_percent(self):
        text = format_download_progress("small", 240_000_000, 480_000_000)
        assert "50%" in text
        assert "240 MB" in text and "480 MB" in text

    def test_gigabytes(self):
        text = format_download_progress("large", 800_000_000, 1_600_000_000)
        assert "1.6 GB" in text


# ──────────────── PREPARING ────────────────


class TestPreparing:
    def test_happy_path_starts_recording(self, tmp_path):
        h = Harness(tmp_path)

        assert h.start() is None

        assert h.state.phase is LivePhase.RECORDING
        assert h.wf.is_active is True
        assert h.spy.recording_started == 1
        assert h.log == ["session.start", "recorder.start"]
        recorder = h.recorders[0]
        assert recorder.device_id == 1 and recorder.dest == tmp_path
        assert any("ライブ" in m and "[1] マイク" in m for m in h.spy.logs)

    def test_preparing_phase_is_reported_before_recording(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)

        assert h.start() is None
        assert h.state.phase is LivePhase.PREPARING
        assert h.wf.is_active is True
        run_pending()
        assert h.state.phase is LivePhase.RECORDING
        phases = [s.phase for s in h.spy.states]
        assert phases.index(LivePhase.PREPARING) < phases.index(LivePhase.RECORDING)

    def test_chunk_sink_feeds_level_meter_and_session(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        chunk = np.full((512, 1), 0.1, dtype=np.float32)

        h.recorders[0].chunk_sink(chunk)

        assert len(h.sessions[0].fed) == 1
        reading = h.wf.poll_meter()
        assert isinstance(reading, MeterReading)
        assert reading.level_db == pytest.approx(-20.0, abs=0.1)
        assert reading.backlog_sec == 1.5
        assert reading.dropped_chunks == 3

    def test_session_receives_engine_vad_backend(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        session = h.sessions[0]
        session.engine.load()
        assert h.engine_api.created[0].loaded is True
        assert session.engine.transcribe(None, "ja") == "faster-whisper:ja"
        assert session.vad.kind == "silero"
        assert isinstance(session.backend, FakeClient)
        assert session.backend.timeout is None  # 本番クライアントは既定タイムアウト

    def test_model_download_success_reports_progress_and_clears_banner(self, tmp_path):
        fetcher = FakeFetcher(cached=False, progress=[(0, 0), (240_000_000, 480_000_000)])
        h = Harness(tmp_path, fetcher=fetcher)

        h.start()

        assert len(fetcher.fetched) == 1
        texts = [b.text for s in h.spy.states for b in s.banners if b.kind == BANNER_MODEL_DOWNLOAD]
        assert any("50%" in t for t in texts)
        assert h.banner(BANNER_MODEL_DOWNLOAD) is None
        assert h.state.phase is LivePhase.RECORDING

    def test_cached_model_is_not_fetched(self, tmp_path):
        fetcher = FakeFetcher(cached=True)
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        assert fetcher.fetched == []
        assert all(b.kind != BANNER_MODEL_DOWNLOAD for s in h.spy.states for b in s.banners)

    def test_progress_updates_are_throttled(self, tmp_path):
        progress = [(i * 1_000_000, 100_000_000) for i in range(1, 50)]
        fetcher = FakeFetcher(cached=False, progress=progress)
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        texts = {b.text for s in h.spy.states for b in s.banners if b.kind == BANNER_MODEL_DOWNLOAD}
        # 時計が進まないので最初の通知以外は間引かれる
        assert len(texts) <= 3

    def test_model_download_failure_does_not_start_recording(self, tmp_path):
        fetcher = FakeFetcher(cached=False, fail=ConnectionError("offline"))
        h = Harness(tmp_path, fetcher=fetcher)

        h.start()

        assert h.recorders == []
        assert h.sessions == []
        assert h.state.phase is LivePhase.IDLE
        assert h.wf.is_active is False
        assert h.spy.recording_started == 0
        assert len(h.spy.errors) == 1
        assert "ネットワーク" in h.spy.errors[0]
        assert h.banner(BANNER_MODEL_DOWNLOAD) is None

    def test_disk_full_message_uses_required_bytes(self, tmp_path):
        import errno

        fetcher = FakeFetcher(
            cached=False,
            fail=OSError(errno.ENOSPC, "No space"),
            progress=[(0, 1_600_000_000)],
        )
        h = Harness(tmp_path, fetcher=fetcher)
        h.start()
        assert "1.6 GB" in h.spy.errors[0]

    def test_unknown_model_is_setup_error(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_whisper_model="unknown-model")
        h = Harness(tmp_path, config=config)
        h.start()
        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert "unknown-model" in h.spy.errors[0]

    def test_mlx_fallback_shows_banner_and_fetches_fallback_model(self, tmp_path):
        config = VoiceNoteConfig(
            save_folder=str(tmp_path), live_engine="mlx", live_whisper_model="large-v3-turbo"
        )
        h = Harness(tmp_path, config=config, unavailable={"mlx"})

        h.start()

        banner = h.banner(BANNER_ENGINE_FALLBACK)
        assert banner is not None and "faster-whisper" in banner.text
        assert h.fetcher.resolved == [("faster-whisper", "small")]
        assert h.state.phase is LivePhase.RECORDING

    def test_engine_unavailable_without_fallback_is_error(self, tmp_path):
        h = Harness(tmp_path, unavailable={"faster-whisper"})
        h.start()
        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert "音声認識エンジン" in h.spy.errors[0]

    def test_load_time_fallback_warning_becomes_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_engine="mlx")
        h = Harness(tmp_path, config=config, load_fails={"mlx"})
        h.start()
        assert h.banner(BANNER_ENGINE_FALLBACK) is None

        h.sessions[0].engine.load()  # asr スレッドでの load() 時に mlx が使えないと判明

        assert "faster-whisper" in h.banner(BANNER_ENGINE_FALLBACK).text
        assert h.sessions[0].engine.transcribe(None, "ja") == "faster-whisper:ja"

    def test_engine_use_is_serialized_across_runs(self, tmp_path):
        """放棄した前回の asr スレッドと次の実行の asr が同時にエンジンを使わない"""
        h = Harness(tmp_path)
        h.start()
        h.wf.stop()
        h.start()
        first, second = h.engine_api.created
        first.load_gate = threading.Event()
        old = threading.Thread(target=h.sessions[0].engine.load, daemon=True)
        new = threading.Thread(target=h.sessions[1].engine.load, daemon=True)
        try:
            old.start()
            assert first.load_entered.wait(timeout=2.0)
            new.start()
            assert not second.load_entered.wait(timeout=0.2)
        finally:
            first.load_gate.set()
            old.join(timeout=2.0)
            new.join(timeout=2.0)
        assert second.load_entered.is_set()
        assert not old.is_alive() and not new.is_alive()

    @pytest.mark.parametrize(
        ("status", "message", "hint"),
        [
            ("server_down", "接続できません。`ollama serve` を起動してください。", "ollama serve"),
            ("model_missing", "`ollama pull qwen2.5:7b` を実行してください。", "ollama pull"),
        ],
    )
    def test_ollama_unavailable_shows_banner_and_records(self, tmp_path, status, message, hint):
        h = Harness(tmp_path, probe=probe_returning(status, message))

        h.start()

        banner = h.banner(BANNER_OLLAMA)
        assert banner is not None
        assert hint in banner.text and "要約なし" in banner.text
        assert h.sessions[0].backend is None
        assert h.state.phase is LivePhase.RECORDING

    def test_invalid_llm_url_shows_banner(self, tmp_path):
        config = VoiceNoteConfig(
            save_folder=str(tmp_path), live_llm_base_url="http://example.com/v1"
        )

        def client_factory(*args, **kwargs):
            raise ValueError("ループバック以外の URL は使えません")

        h = Harness(tmp_path, config=config)
        h.wf._deps = replace(h.wf._deps, client_factory=client_factory)
        h.start()
        assert "ループバック" in h.banner(BANNER_OLLAMA).text
        assert h.sessions[0].backend is None

    def test_llm_disabled_has_no_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), live_llm_enabled=False)
        h = Harness(tmp_path, config=config)
        h.start()
        assert h.banner(BANNER_OLLAMA) is None
        assert h.sessions[0].backend is None

    def test_vad_fallback_shows_banner_and_records(self, tmp_path):
        h = Harness(tmp_path, vad_fallback=True)
        h.start()
        assert "energy" in h.banner(BANNER_VAD_FALLBACK).text
        assert h.state.phase is LivePhase.RECORDING

    def test_external_send_banner(self, tmp_path):
        config = VoiceNoteConfig(save_folder=str(tmp_path), transcription_mode="openai")
        h = Harness(tmp_path, config=config)
        h.start()
        assert "OpenAI" in h.banner(BANNER_EXTERNAL_SEND).text

    def test_no_external_send_banner_for_local(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        assert h.banner(BANNER_EXTERNAL_SEND) is None

    def test_recorder_start_failure_releases_session(self, tmp_path):
        h = Harness(tmp_path, fail_start=True)

        h.start()

        assert h.state.phase is LivePhase.IDLE
        assert h.wf.is_active is False
        assert h.sessions[0].stop_timeouts  # 起動済みセッションを解放する
        assert "デバイスが使用できません" in h.spy.errors[0]
        assert h.wf.poll_meter() is None

    def test_unexpected_exception_returns_to_idle(self, tmp_path):
        h = Harness(tmp_path, session_factory_error=MemoryError("boom"))
        h.start()
        assert h.state.phase is LivePhase.IDLE
        assert h.recorders == []
        assert len(h.spy.errors) == 1

    def test_restart_after_failure_is_allowed(self, tmp_path):
        h = Harness(tmp_path, fail_start=True)
        h.start()
        h.fail_start = False
        assert h.start() is None
        assert h.state.phase is LivePhase.RECORDING

    def test_new_run_resets_view_state(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "前回")
        h.wf.stop()
        h.start()
        assert h.state.segments == ()

    def test_thread_creation_failure_returns_error(self, tmp_path):
        def failing(*args, **kwargs):
            raise RuntimeError("スレッドを生成できません")

        h = Harness(tmp_path, thread_factory=failing)
        error = h.start()
        assert error is not None
        assert h.state.phase is LivePhase.IDLE


class TestPreparingCancel:
    def test_stop_during_preparing_cancels_without_recording(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()

        h.wf.stop()
        run_pending()

        assert h.recorders == []
        assert h.state.phase is LivePhase.IDLE
        assert h.spy.recording_started == 0
        assert h.spy.errors == []
        assert any("取り消" in m for m in h.spy.logs)

    def test_cancel_after_session_start_releases_session(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()
        original = h._make_session

        def make_and_cancel(*args, **kwargs):
            session = original(*args, **kwargs)
            h.wf.stop()  # セッション生成直後に取り消し
            return session

        h.wf._deps = replace(h.wf._deps, session_factory=make_and_cancel)
        run_pending()

        assert h.recorders == []
        assert h.sessions[0].stop_timeouts
        assert h.state.phase is LivePhase.IDLE

    def test_shutdown_during_preparing_never_starts_recorder(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()

        h.wf.shutdown()
        run_pending()

        assert h.recorders == []
        assert h.saved_notes == []


# ──────────────── 停止 ────────────────


class TestStop:
    def test_stop_order_and_note_saved(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.log.clear()

        h.wf.stop()

        assert h.log == [
            "finalize",
            "on_wav_saved",
            "stop_with_timeout",
            "session.stop",
            "save",
            "on_done",
        ]
        assert h.sessions[0].stop_timeouts == [120.0]
        folder, body, audio = h.saved_notes[0]
        assert folder == tmp_path / "notes"
        assert audio == h.wav
        assert "こんにちは" in body and "要約" in body
        assert h.spy.done == [(h.wav, tmp_path / "notes" / "note0_live.md")]
        assert h.spy.wav_saved == [h.wav]
        assert h.state.phase is LivePhase.IDLE
        assert h.wf.is_active is False

    def test_phases_progress_in_order(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.spy.states.clear()
        h.wf.stop()
        phases = []
        for s in h.spy.states:
            if not phases or phases[-1] is not s.phase:
                phases.append(s.phase)
        assert phases == [
            LivePhase.FINALIZING,
            LivePhase.DRAINING,
            LivePhase.SAVING,
            LivePhase.IDLE,
        ]

    def test_stop_does_not_block_caller(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.wf._thread_factory = DeferredThread

        h.wf.stop()

        assert h.state.phase is LivePhase.FINALIZING
        assert h.recorders[0].finalize_timeouts == []
        run_pending()
        assert h.state.phase is LivePhase.IDLE

    def test_session_stop_runs_in_separate_thread_after_wav(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.wf._thread_factory = DeferredThread
        h.wf.stop()
        DeferredThread.instances[0].run()  # 停止ワーカーだけ実行

        assert h.spy.wav_saved == [h.wav]
        assert h.recorders[0].stop_called is True
        assert h.sessions[0].stop_timeouts == []  # ドレインはまだ
        assert h.state.phase is LivePhase.DRAINING
        assert h.wf.is_active is True

    def test_external_send_warning_in_note(self, tmp_path):
        # send_warning は欠落がある場合の品質注記に併記される
        config = VoiceNoteConfig(save_folder=str(tmp_path / "notes"), format_mode="llm")
        result = LiveResult(segments=(), final_summary="", dropped_chunks=1)
        h = Harness(tmp_path, config=config, session_result=result)
        h.start()
        h.wf.stop()
        assert "OpenAI" in h.saved_notes[0][1]

    def test_no_audio_reports_error_and_releases(self, tmp_path):
        h = Harness(tmp_path, finalized=None)
        h.start()

        h.wf.stop()

        assert h.recorders[0].stop_called is True
        assert h.sessions[0].stop_timeouts == [2.0]
        assert h.saved_notes == []
        assert h.spy.done == []
        assert any("録音データがありません" in e for e in h.spy.errors)
        assert h.state.phase is LivePhase.IDLE

    def test_session_stop_exception_saves_partial_note(self, tmp_path):
        h = Harness(tmp_path, session_fail_stop=True)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "集めた区間")
        h.wf.stop()
        body = h.saved_notes[0][1]
        assert "集めた区間" in body
        assert "未処理あり: はい" in body
        assert h.spy.done

    def test_note_save_failure_reports_wav_location(self, tmp_path):
        def failing_saver(folder, body, audio_file=None):
            raise RuntimeError("保存エラー: disk")

        h = Harness(tmp_path, note_saver=failing_saver)
        h.start()
        h.wf.stop()
        assert h.spy.done == []
        assert "保存エラー" in h.spy.errors[0]
        assert str(h.wav) in h.spy.errors[0]
        assert h.state.phase is LivePhase.IDLE

    def test_stop_when_idle_is_noop(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.stop()
        assert h.log == []

    def test_stop_thread_creation_failure_reports_error(self, tmp_path):
        h = Harness(tmp_path)
        h.start()

        def failing(*args, **kwargs):
            raise RuntimeError("スレッドを生成できません")

        h.wf._thread_factory = failing
        h.wf.stop()
        assert h.spy.errors
        assert h.state.phase is LivePhase.IDLE


class TestStreamCloseTracking:
    def test_timed_out_stream_blocks_restart_until_closed(self, tmp_path):
        h = Harness(tmp_path, stop_result=False)
        h.start()
        h.wf.stop()
        h.recorders[0].closing = True

        error = h.start()
        assert error is not None and "解放" in error

        h.recorders[0].closing = False
        assert h.start() is None

    def test_gives_up_waiting_after_grace_period(self, tmp_path):
        h = Harness(tmp_path, stop_result=False)
        h.start()
        h.wf.stop()
        h.recorders[0].closing = True
        h.clock_now += 1000.0

        assert h.start() is None
        assert any("解放されない" in m for m in h.spy.logs)


# ──────────────── ドレイン中断 ────────────────


class TestAbortDrain:
    def _draining(self, tmp_path, **kwargs) -> Harness:
        h = Harness(tmp_path, **kwargs)
        h.start()
        session = h.sessions[0]
        session.callbacks.on_segment(0, 0.0, 1.0, "途中まで")
        session.callbacks.on_summary("途中の要約")
        h.wf._thread_factory = DeferredThread
        h.wf.stop()
        DeferredThread.instances[0].run()  # 停止ワーカー(WAV確定)だけ
        assert h.state.phase is LivePhase.DRAINING
        h.wf._thread_factory = ImmediateThread
        return h

    def test_abort_saves_partial_note_with_incomplete_flag(self, tmp_path):
        h = self._draining(tmp_path)

        assert h.wf.abort_drain() is True

        folder, body, audio = h.saved_notes[0]
        assert audio == h.wav
        assert "途中まで" in body and "途中の要約" in body
        assert "未処理あり: はい" in body
        assert "破棄チャンク: 3" in body
        assert h.spy.done == [(h.wav, folder / "note0_live.md")]
        assert h.state.phase is LivePhase.IDLE

    def test_late_drain_result_is_ignored(self, tmp_path):
        h = self._draining(tmp_path)
        h.wf.abort_drain()

        run_pending()  # 放棄したドレインスレッドが後から戻る

        assert len(h.saved_notes) == 1
        assert len(h.spy.done) == 1

    def test_late_session_callbacks_are_ignored(self, tmp_path):
        h = self._draining(tmp_path)
        callbacks = h.sessions[0].callbacks
        h.wf.abort_drain()
        before = h.state

        callbacks.on_segment(1, 1.0, 2.0, "遅れて届いた")
        callbacks.on_summary("遅れた要約")

        assert h.state == before

    def test_restart_rejected_while_abandoned_drain_alive(self, tmp_path):
        h = self._draining(tmp_path)
        h.wf.abort_drain()

        assert h.wf.is_active is True
        error = h.start()
        assert error is not None and "後処理" in error

        run_pending()
        assert h.wf.is_active is False
        assert h.start() is None

    def test_abort_when_not_draining_returns_false(self, tmp_path):
        h = Harness(tmp_path)
        assert h.wf.abort_drain() is False
        h.start()
        assert h.wf.abort_drain() is False


# ──────────────── 排他・再入 ────────────────


class TestExclusion:
    def test_can_start_false_rejects(self, tmp_path):
        h = Harness(tmp_path, can_start=lambda: False)
        error = h.start()
        assert error is not None and "実行中" in error
        assert h.sessions == []
        assert h.state.phase is LivePhase.IDLE

    def test_double_start_rejected(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        error = h.start()
        assert error is not None and "既に" in error
        assert len(h.sessions) == 1

    def test_start_after_shutdown_rejected(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.shutdown()
        assert h.start() is not None


# ──────────────── 表示・メーター ────────────────


class TestDisplay:
    def test_session_callbacks_update_state(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        cb = h.sessions[0].callbacks

        cb.on_segment(0, 0.0, 1.5, "こんにちは")
        cb.on_summary("要約です")
        cb.on_status("ライブ文字起こし中")
        cb.on_warning("区間 1.0s の文字起こしに失敗しました: boom")

        assert h.state.segments == (LiveSegment(0, 0.0, 1.5, "こんにちは"),)
        assert h.state.summary == "要約です"
        assert h.state.status == "ライブ文字起こし中"
        assert h.state.warnings == ("区間 1.0s の文字起こしに失敗しました: boom",)
        assert h.spy.states[-1] == h.state

    def test_warning_throttle_uses_injected_clock(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        cb = h.sessions[0].callbacks
        cb.on_warning("区間 1.0s の文字起こしに失敗しました: a")
        cb.on_warning("区間 2.0s の文字起こしに失敗しました: b")
        assert len(h.state.warnings) == 1
        h.clock_now += 11.0
        cb.on_warning("区間 3.0s の文字起こしに失敗しました: c")
        assert len(h.state.warnings) == 2

    def test_on_state_not_called_when_unchanged(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        cb = h.sessions[0].callbacks
        cb.on_summary("同じ")
        count = len(h.spy.states)
        cb.on_summary("同じ")
        assert len(h.spy.states) == count

    def test_callback_errors_do_not_break_workflow(self, tmp_path):
        h = Harness(tmp_path)

        def boom(state):
            raise RuntimeError("UI 側の失敗")

        h.wf._callbacks = replace(h.wf._callbacks, on_state=boom)
        assert h.start() is None
        assert h.wf.phase is LivePhase.RECORDING

    def test_poll_meter_none_when_not_recording(self, tmp_path):
        h = Harness(tmp_path)
        assert h.wf.poll_meter() is None

    def test_poll_meter_pending_assessment_initially(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        reading = h.wf.poll_meter()
        assert reading.input_level.status == "pending"
        assert h.banner(BANNER_LOW_INPUT) is None

    def test_low_input_banner_added_and_cleared(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        sink = h.recorders[0].chunk_sink
        quiet = np.full((512, 1), 0.001, dtype=np.float32)  # -60 dB
        for _ in range(30):
            sink(quiet)

        reading = h.wf.poll_meter()
        assert reading.input_level.is_low
        assert "入力が小さい" in h.banner(BANNER_LOW_INPUT).text

        loud = np.full((512, 1), 0.1, dtype=np.float32)  # -20 dB
        for _ in range(500):
            sink(loud)
        h.wf.poll_meter()
        assert h.banner(BANNER_LOW_INPUT) is None


# ──────────────── shutdown ────────────────


class TestShutdown:
    def test_shutdown_while_recording_finalizes_and_saves_partial_note(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "閉じる前")
        h.log.clear()

        h.wf.shutdown()

        recorder = h.recorders[0]
        assert recorder.finalize_timeouts == [1.0]
        assert h.log[0] == "finalize"
        assert h.log.index("save") < h.log.index("stop_with_timeout")
        body = h.saved_notes[0][1]
        assert "閉じる前" in body and "未処理あり: はい" in body
        assert recorder.stop_called is True

    def test_shutdown_while_draining_saves_partial_note(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "ドレイン中")
        h.wf._thread_factory = DeferredThread
        h.wf.stop()
        DeferredThread.instances[0].run()
        h.wf._thread_factory = ImmediateThread

        h.wf.shutdown()

        assert "ドレイン中" in h.saved_notes[0][1]
        run_pending()
        assert len(h.saved_notes) == 1

    def test_shutdown_while_stream_stop_hangs_saves_partial_note(self, tmp_path):
        """FINALIZING(WAV確定後・ストリーム停止のハング中)に閉じても部分ノートを1つだけ残す"""
        h = Harness(tmp_path)
        h.start()
        h.sessions[0].callbacks.on_segment(0, 0.0, 1.0, "停止待ち")
        h.recorders[0].on_stop = h.wf.shutdown

        h.wf.stop()

        assert len(h.saved_notes) == 1
        body = h.saved_notes[0][1]
        assert "停止待ち" in body and "未処理あり: はい" in body
        assert h.spy.done == []
        assert h.sessions[0].stop_timeouts == []  # ドレインは始まらない
        assert h.state.phase is LivePhase.IDLE

    def test_shutdown_without_audio_saves_nothing(self, tmp_path):
        h = Harness(tmp_path, finalized=None)
        h.start()
        h.wf.shutdown()
        assert h.saved_notes == []
        assert h.recorders[0].stop_called is True

    def test_shutdown_when_idle_is_noop(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.shutdown()
        assert h.log == []

    def test_shutdown_survives_save_failure(self, tmp_path):
        def failing_saver(folder, body, audio_file=None):
            raise RuntimeError("保存エラー")

        h = Harness(tmp_path, note_saver=failing_saver)
        h.start()
        h.wf.shutdown()  # 例外を送出しないこと
        assert h.recorders[0].stop_called is True


class TestUpdateConfig:
    def test_running_session_keeps_start_time_config(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.wf.update_config(VoiceNoteConfig(save_folder=str(tmp_path / "other")))
        h.wf.stop()
        assert h.saved_notes[0][0] == tmp_path / "notes"

    def test_next_run_uses_new_config(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_config(VoiceNoteConfig(save_folder=str(tmp_path / "other")))
        h.start()
        h.wf.stop()
        assert h.saved_notes[0][0] == tmp_path / "other"
