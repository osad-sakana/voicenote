"""gui/live_workflow.py のテスト用 Fake 群と Harness(実ウィンドウ・実音声・実ネットワークなし)

録音・セッション・エンジン・VAD・LLM・モデル取得はすべて Fake に差し替え、
スレッドは Immediate(同期実行)/ Deferred(捕捉のみ)で制御する。
DeferredThread.instances は各テストモジュールの autouse fixture から
reset_deferred_threads() で初期化する。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config import VoiceNoteConfig
from gui.live_workflow import LiveDeps, LiveWorkflow, LiveWorkflowCallbacks, ModelFetcher
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


def reset_deferred_threads() -> None:
    """捕捉済みの Deferred スレッドを忘れる(テストごとに呼ぶ)"""
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
