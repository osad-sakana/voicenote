"""live.session (LiveSession / LiveSettings) の統合テスト。

実エンジン・実LLMは使わず、FakeEngine / FakeBackend と合成音声(サイン波バースト+無音)で
スレッド構成全体(Framer→VAD→Segmenter→ASR→要約)を検証する。
sleep に依存しないよう、待機は Event / 短い timeout 付きのポーリングで行う。
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from config import VoiceNoteConfig
from live.session import LiveCallbacks, LiveResult, LiveSegment, LiveSession, LiveSettings
from live.vad import EnergyVAD

SR = 16000
CHUNK = 1600  # 100ms
WAIT_SEC = 5.0
SUMMARY_TEXT = "現在の話題: テスト\n進捗:\n- 要点その1が話された"


# ---------------------------------------------------------------- helpers


def tone(sec: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def bursts(n: int, speech_sec: float = 1.0, gap_sec: float = 1.0) -> np.ndarray:
    parts = []
    for _ in range(n):
        parts.extend([tone(speech_sec), silence(gap_sec)])
    return np.concatenate(parts)


def feed_audio(session: LiveSession, audio: np.ndarray) -> None:
    for i in range(0, len(audio), CHUNK):
        session.feed(audio[i : i + CHUNK].reshape(-1, 1))


def wait_until(predicate, timeout: float = WAIT_SEC) -> bool:
    deadline = time.monotonic() + timeout
    tick = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        tick.wait(0.01)
    return predicate()


@dataclass(frozen=True)
class Text:
    """TranscribedText 互換の最小オブジェクト"""

    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    language: str | None = None


class FakeEngine:
    """呼び出し回数に応じた応答を返すエンジン。

    responses の要素: str(そのテキスト) / Exception(送出) / threading.Event(set まで待機)
    responses を使い切ったら `発話{n}` を返す。
    """

    def __init__(self, responses=(), load_error: Exception | None = None):
        self._responses = list(responses)
        self._load_error = load_error
        self.calls = 0
        self.languages: list[str | None] = []
        self.load_thread: str | None = None

    def load(self) -> None:
        self.load_thread = threading.current_thread().name
        if self._load_error is not None:
            raise self._load_error

    def transcribe(self, samples, language):
        self.calls += 1
        self.languages.append(language)
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, threading.Event):
                item.wait()
                return Text(f"発話{self.calls}")
            if isinstance(item, Exception):
                raise item
            return Text(item)
        return Text(f"発話{self.calls}")


class FakeBackend:
    def __init__(self, reply: str = SUMMARY_TEXT, error: Exception | None = None, gate=None):
        self.reply = reply
        self.error = error
        self.gate = gate
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.reply

    def list_models(self):
        return ["qwen2.5:7b"]


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Recorder:
    """コールバックの受信内容を記録する"""

    def __init__(self):
        self.segments: list[tuple[int, float, float, str]] = []
        self.summaries: list[str] = []
        self.warnings: list[str] = []
        self.statuses: list[str] = []

    def callbacks(self) -> LiveCallbacks:
        return LiveCallbacks(
            on_segment=lambda i, s, e, t: self.segments.append((i, s, e, t)),
            on_summary=self.summaries.append,
            on_status=self.statuses.append,
            on_warning=self.warnings.append,
        )


def make_settings(**overrides) -> LiveSettings:
    base = {"summary_poll_sec": 0.01, "summary_interval_sec": 45}
    return LiveSettings(**{**base, **overrides})


@pytest.fixture
def gates():
    """テスト終了時に必ず解放する Event 群(daemon スレッドを残さないため)"""
    created: list[threading.Event] = []

    def factory() -> threading.Event:
        event = threading.Event()
        created.append(event)
        return event

    yield factory
    for event in created:
        event.set()


def make_session(engine, backend=None, rec=None, clock=None, **settings) -> LiveSession:
    rec = rec or Recorder()
    return LiveSession(
        make_settings(**settings),
        engine=engine,
        vad=EnergyVAD(),
        backend=backend,
        callbacks=rec.callbacks(),
        clock=clock or FakeClock(),
    )


# ---------------------------------------------------------------- LiveSettings


class TestLiveSettings:
    def test_from_default_config(self):
        settings = LiveSettings.from_config(VoiceNoteConfig())
        assert settings.engine == "faster-whisper"
        assert settings.model == "small"
        assert settings.language is None
        assert settings.vad == "energy"
        assert settings.llm_enabled is True
        assert settings.llm_base_url == "http://127.0.0.1:11434/v1"
        assert settings.llm_model == "qwen2.5:7b"
        assert settings.summary_interval_sec == 45

    def test_language_is_passed_through(self):
        settings = LiveSettings.from_config(VoiceNoteConfig(live_language="ja"))
        assert settings.language == "ja"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"live_engine": "whisper.cpp"},
            {"live_vad": "webrtc"},
            {"live_summary_interval_sec": 0},
            {"live_summary_interval_sec": -5},
            {"live_summary_interval_sec": "45"},
            {"live_summary_interval_sec": True},
            {"live_language": None},
            {"live_whisper_model": ""},
            {"live_llm_enabled": "yes"},
        ],
    )
    def test_invalid_values_raise(self, overrides):
        with pytest.raises(ValueError):
            LiveSettings.from_config(VoiceNoteConfig(**overrides))

    def test_is_frozen(self):
        settings = LiveSettings()
        with pytest.raises(AttributeError):
            settings.engine = "mlx"  # type: ignore[misc]

    def test_callbacks_default_to_noop(self):
        callbacks = LiveCallbacks()
        callbacks.on_segment(0, 0.0, 1.0, "x")
        callbacks.on_summary("s")
        callbacks.on_status("st")
        callbacks.on_warning("w")


# ---------------------------------------------------------------- LiveSession


class TestSegmentsFlow:
    def test_segments_are_emitted_in_order(self):
        rec = Recorder()
        engine = FakeEngine()
        session = make_session(engine, rec=rec)
        session.start()
        feed_audio(session, bursts(3))
        assert wait_until(lambda: len(rec.segments) == 3)
        result = session.stop(timeout=WAIT_SEC)

        assert [s[0] for s in rec.segments] == [0, 1, 2]
        assert [s[3] for s in rec.segments] == ["発話1", "発話2", "発話3"]
        starts = [s[1] for s in rec.segments]
        assert starts == sorted(starts)
        assert all(s[1] < s[2] for s in rec.segments)
        assert isinstance(result, LiveResult)
        assert result.segments == tuple(LiveSegment(*s) for s in rec.segments)
        assert result.complete is True

    def test_stop_flushes_open_segment(self):
        rec = Recorder()
        session = make_session(FakeEngine(), rec=rec)
        session.start()
        feed_audio(session, tone(1.0))  # 末尾に無音が無いので区間は開いたまま
        result = session.stop(timeout=WAIT_SEC)
        assert len(result.segments) == 1
        assert result.segments[0].text == "発話1"

    def test_engine_load_runs_on_asr_thread(self):
        engine = FakeEngine()
        session = make_session(engine)
        session.start()
        session.stop(timeout=WAIT_SEC)
        assert engine.load_thread is not None
        assert engine.load_thread != threading.current_thread().name

    def test_language_is_passed_to_engine(self):
        engine = FakeEngine()
        session = make_session(engine, language="ja")
        session.start()
        feed_audio(session, bursts(1))
        session.stop(timeout=WAIT_SEC)
        assert engine.languages == ["ja"]

    def test_threads_are_daemon(self):
        session = make_session(FakeEngine(), backend=FakeBackend())
        before = set(threading.enumerate())
        session.start()
        try:
            started = [t for t in threading.enumerate() if t not in before]
            names = {t.name for t in started}
            assert {"live-segmenter", "live-asr", "live-summary"} <= names
            assert all(t.daemon for t in started if t.name.startswith("live-"))
        finally:
            session.stop(timeout=WAIT_SEC)


class TestFiltering:
    def test_hallucination_is_dropped(self):
        rec = Recorder()
        engine = FakeEngine(responses=["ご視聴ありがとうございました", "本題に入ります"])
        session = make_session(engine, rec=rec)
        session.start()
        feed_audio(session, bursts(2))
        result = session.stop(timeout=WAIT_SEC)
        assert [s.text for s in result.segments] == ["本題に入ります"]
        assert result.segments[0].index == 0

    def test_no_speech_prob_is_respected(self):
        class NoSpeechEngine(FakeEngine):
            def transcribe(self, samples, language):
                return Text("それっぽい文章", no_speech_prob=0.9, avg_logprob=-1.5)

        session = make_session(NoSpeechEngine())
        session.start()
        feed_audio(session, bursts(1))
        assert session.stop(timeout=WAIT_SEC).segments == ()


class TestResilience:
    def test_engine_exception_does_not_kill_session(self):
        rec = Recorder()
        engine = FakeEngine(responses=[RuntimeError("boom"), "続きの発話"])
        session = make_session(engine, rec=rec)
        session.start()
        feed_audio(session, bursts(2))
        result = session.stop(timeout=WAIT_SEC)
        assert [s.text for s in result.segments] == ["続きの発話"]
        assert any("boom" in w for w in rec.warnings)

    def test_engine_load_failure_warns_once_and_stop_returns(self):
        rec = Recorder()
        engine = FakeEngine(load_error=RuntimeError("no model"))
        session = make_session(engine, rec=rec)
        session.start()
        feed_audio(session, bursts(3))
        result = session.stop(timeout=WAIT_SEC)
        assert result.segments == ()
        assert engine.calls == 0
        assert len([w for w in rec.warnings if "no model" in w]) == 1

    def test_callback_exception_does_not_kill_session(self):
        received: list[str] = []

        def on_segment(i, s, e, text):
            received.append(text)
            raise RuntimeError("ui bug")

        session = LiveSession(
            make_settings(),
            engine=FakeEngine(),
            vad=EnergyVAD(),
            backend=None,
            callbacks=LiveCallbacks(on_segment=on_segment),
        )
        session.start()
        feed_audio(session, bursts(2))
        result = session.stop(timeout=WAIT_SEC)
        assert len(result.segments) == 2
        assert received == ["発話1", "発話2"]

    def test_hanging_engine_stop_returns_within_timeout(self, gates):
        rec = Recorder()
        engine = FakeEngine(responses=[gates()])
        session = make_session(engine, rec=rec)
        session.start()
        feed_audio(session, bursts(2))
        assert wait_until(lambda: engine.calls == 1)

        began = time.monotonic()
        result = session.stop(timeout=0.3)
        elapsed = time.monotonic() - began

        assert elapsed < 2.0
        assert result.complete is False
        assert rec.warnings  # 未処理区間がある旨を警告

    def test_hanging_llm_final_summary_falls_back(self, gates):
        backend = FakeBackend(gate=gates())
        session = make_session(FakeEngine(), backend=backend)
        session.start()
        feed_audio(session, bursts(1))
        began = time.monotonic()
        result = session.stop(timeout=0.5)
        assert time.monotonic() - began < 2.0
        assert result.final_summary == ""


class TestBackpressure:
    def test_feed_only_enqueues_and_counts_drops(self):
        session = make_session(FakeEngine(), chunk_queue_size=2)
        # start していないので消費されず、3個目以降は破棄される
        for _ in range(5):
            session.feed(np.zeros((CHUNK, 1), dtype=np.float32))
        assert session.dropped_chunks == 3
        result = session.stop(timeout=WAIT_SEC)
        assert result.dropped_chunks == 3

    def test_feed_after_stop_is_ignored(self):
        session = make_session(FakeEngine())
        session.start()
        session.stop(timeout=WAIT_SEC)
        session.feed(np.zeros((CHUNK, 1), dtype=np.float32))
        assert session.dropped_chunks == 0

    def test_segment_queue_drops_oldest_with_warning(self, gates):
        rec = Recorder()
        gate = gates()
        engine = FakeEngine(responses=[gate])
        session = make_session(engine, rec=rec, segment_queue_size=1)
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: engine.calls == 1)  # 1区間目で詰まらせる
        feed_audio(session, bursts(3))  # キュー容量1 → 2つ破棄
        assert wait_until(lambda: len(rec.warnings) >= 2)
        gate.set()
        result = session.stop(timeout=WAIT_SEC)
        assert [s.text for s in result.segments] == ["発話1", "発話2"]
        assert result.dropped_segments == 2
        # 最古を捨てて最新を残す: 残った区間は最後の発話
        assert result.segments[1].start_sec > 5.0


class TestSummary:
    def test_slow_engine_increases_backlog_and_holds_summary(self, gates):
        rec = Recorder()
        clock = FakeClock(0.0)
        gate = gates()
        backend = FakeBackend()
        # 1区間目で初回要約を済ませ、2区間目で未要約テキストを作り、3区間目で詰まらせる
        engine = FakeEngine(responses=["最初の発話です", "二番目の発話です", gate])
        session = make_session(engine, backend=backend, rec=rec, clock=clock)
        session.start()
        feed_audio(session, bursts(1))
        # 初回要約(last_run=None)は即時に走るので、それが終わるのを待つ
        assert wait_until(lambda: len(rec.summaries) == 1)
        baseline_calls = backend.calls
        feed_audio(session, bursts(1))
        assert wait_until(lambda: len(rec.segments) == 2)  # clock=0 なので要約はまだ走らない

        feed_audio(session, bursts(3, speech_sec=4.0))
        assert wait_until(lambda: session.backlog_sec > 10.0)
        clock.now = 1000.0  # interval を十分超過させる
        # 詰まっている間は要約が走らない
        assert not wait_until(lambda: backend.calls > baseline_calls, timeout=0.3)

        gate.set()
        assert wait_until(lambda: session.backlog_sec == 0.0)
        clock.now = 2000.0
        assert wait_until(lambda: backend.calls > baseline_calls)
        session.stop(timeout=WAIT_SEC)
        assert rec.summaries

    def test_final_summary_runs_on_stop(self):
        rec = Recorder()
        backend = FakeBackend()
        clock = FakeClock(0.0)
        engine = FakeEngine()
        session = make_session(engine, backend=backend, rec=rec, clock=clock)
        # 周期要約が走るかどうかに依らず、stop 時の最終要約が結果に入ることを検証する
        session.start()
        feed_audio(session, bursts(1))
        result = session.stop(timeout=WAIT_SEC)
        assert result.final_summary == SUMMARY_TEXT
        assert rec.summaries[-1] == SUMMARY_TEXT

    def test_final_summary_failure_keeps_previous(self):
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend)
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: backend.calls >= 1)
        backend.error = RuntimeError("llm down")
        feed_audio(session, bursts(1))
        result = session.stop(timeout=WAIT_SEC)
        assert result.final_summary == SUMMARY_TEXT

    def test_works_without_backend(self):
        rec = Recorder()
        session = make_session(FakeEngine(), backend=None, rec=rec)
        session.start()
        feed_audio(session, bursts(2))
        result = session.stop(timeout=WAIT_SEC)
        assert len(result.segments) == 2
        assert result.final_summary == ""
        assert rec.summaries == []

    def test_stop_is_idempotent(self):
        session = make_session(FakeEngine())
        session.start()
        feed_audio(session, bursts(1))
        first = session.stop(timeout=WAIT_SEC)
        assert session.stop(timeout=WAIT_SEC) == first

    def test_start_twice_raises(self):
        session = make_session(FakeEngine())
        session.start()
        try:
            with pytest.raises(RuntimeError):
                session.start()
        finally:
            session.stop(timeout=WAIT_SEC)
