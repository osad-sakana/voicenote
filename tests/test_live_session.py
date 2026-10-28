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

    def transcribe(self, samples, language, prompt=None):
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
        assert settings.vad == "silero"
        assert settings.llm_enabled is True
        assert settings.llm_base_url == "http://127.0.0.1:11434/v1"
        assert settings.llm_model == "qwen2.5:7b"
        assert settings.summary_interval_sec == 45

    def test_refine_defaults(self):
        settings = LiveSettings.from_config(VoiceNoteConfig())
        assert settings.refine_enabled is False
        assert settings.refine_model == "large-v3"

    def test_refine_values_from_config(self):
        config = VoiceNoteConfig(live_refine_enabled=True, live_refine_model="medium")
        settings = LiveSettings.from_config(config)
        assert settings.refine_enabled is True
        assert settings.refine_model == "medium"

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
            {"live_refine_enabled": "yes"},
            {"live_refine_enabled": 1},
            {"live_refine_model": ""},
            {"live_refine_model": "  "},
            {"live_refine_model": None},
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
            def transcribe(self, samples, language, prompt=None):
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

    def test_vad_exception_warns_once_and_session_survives(self):
        """VAD の例外ではセッションを止めず、警告は1回だけ出す"""

        class FailingVAD:
            def speech_prob(self, frame):
                raise RuntimeError("vad boom")

        rec = Recorder()
        session = LiveSession(
            make_settings(),
            engine=FakeEngine(),
            vad=FailingVAD(),
            backend=None,
            callbacks=rec.callbacks(),
            clock=FakeClock(),
        )
        session.start()
        feed_audio(session, bursts(2))
        result = session.stop(timeout=WAIT_SEC)
        vad_warnings = [w for w in rec.warnings if "音声区間の検出に失敗しました" in w]
        assert len(vad_warnings) == 1
        assert "vad boom" in vad_warnings[0]
        assert result.segments == ()
        assert result.complete is True

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
        # 最終要約の上限は timeout(ドレイン)とは別。ここでは短い上限を明示する
        result = session.stop(timeout=0.5, summary_timeout=0.5)
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

    def test_dropped_chunks_shift_later_timestamps(self):
        """破棄されたチャンクのサンプル数だけ、以降の区間の時刻が後ろにずれる(WAVと一致)"""
        chunk = 1024  # フレーム長(512)の倍数にして Framer の端数を出さない
        head = [tone(1.024), silence(0.768)]  # 1区間目を閉じ切ってから破棄を起こす
        tail = [silence(0.512), tone(1.024)]
        dropped = np.zeros(chunk, dtype=np.float32)

        def run(drop: bool) -> tuple[LiveSegment, ...]:
            head_audio = np.concatenate(head)
            queue_size = len(head_audio) // chunk
            session = make_session(FakeEngine(), chunk_queue_size=queue_size if drop else 2000)
            for i in range(0, len(head_audio), chunk):
                session.feed(head_audio[i : i + chunk].reshape(-1, 1))
            if drop:
                session.feed(dropped.reshape(-1, 1))  # 満杯のため破棄される
                assert session.dropped_chunks == 1
            session.start()
            assert wait_until(lambda: session._chunks.qsize() == 0)
            tail_audio = np.concatenate(tail)
            for i in range(0, len(tail_audio), chunk):
                session.feed(tail_audio[i : i + chunk].reshape(-1, 1))
            return session.stop(timeout=WAIT_SEC).segments

        expected = run(drop=False)
        actual = run(drop=True)
        assert len(actual) == len(expected) == 2
        assert actual[0].start_sec == pytest.approx(expected[0].start_sec)
        assert actual[1].start_sec == pytest.approx(expected[1].start_sec + chunk / SR)

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

    def test_final_summary_update_is_notified(self):
        """最終要約で内容が変わったら on_summary で通知し、結果にも入る"""
        rec = Recorder()
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: rec.summaries == [SUMMARY_TEXT])
        updated = f"{SUMMARY_TEXT}\n- 要点その2が話された"
        backend.reply = updated
        feed_audio(session, bursts(1))
        assert wait_until(lambda: len(rec.segments) == 2)  # clock=0 なので周期要約は走らない
        result = session.stop(timeout=WAIT_SEC)
        assert result.final_summary == updated
        assert rec.summaries == [SUMMARY_TEXT, updated]

    def test_final_summary_timeout_keeps_previous_with_warning(self, gates):
        """最終要約が timeout 内に終わらなければ直前の要約を使い、警告する"""
        rec = Recorder()
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: rec.summaries == [SUMMARY_TEXT])
        backend.gate = gates()  # 以降の要約呼び出しは解放されるまで返らない
        feed_audio(session, bursts(1))
        assert wait_until(lambda: len(rec.segments) == 2)
        began = time.monotonic()
        result = session.stop(timeout=0.5, summary_timeout=0.5)
        assert time.monotonic() - began < 2.0
        assert result.final_summary == SUMMARY_TEXT
        assert any("最終要約が時間内に終わらなかった" in w for w in rec.warnings)

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

    def test_final_summary_is_built_from_all_segments(self):
        """停止時の最終要約は、ローリング要約ではなく全セグメントの文字起こしから作る"""
        seen: list[str] = []

        class RecordingBackend(FakeBackend):
            def complete(self, messages):
                seen.append("\n".join(m["content"] for m in messages))
                return super().complete(messages)

        engine = FakeEngine(responses=["最初の話題です", "二番目の話題です", "三番目の話題です"])
        session = make_session(
            engine, backend=RecordingBackend(), clock=FakeClock(0.0), rec=Recorder()
        )
        session.start()
        feed_audio(session, bursts(3))
        result = session.stop(timeout=WAIT_SEC)
        assert len(result.segments) == 3
        assert result.final_summary == SUMMARY_TEXT
        final_prompt = seen[-1]
        assert all(
            t in final_prompt for t in ("最初の話題です", "二番目の話題です", "三番目の話題です")
        )

    def test_full_summary_failure_falls_back_to_rolling(self):
        """全文要約が不採用でも、ローリング要約で最終要約を試みる"""
        backend = FakeBackend(reply="short")  # 日本語として不採用になり、全文要約は None
        rec = Recorder()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        result = session.stop(timeout=WAIT_SEC)
        assert backend.calls >= 2  # 全文要約 + ローリング要約のフォールバック
        assert result.final_summary == ""  # どちらも不採用なので、直前の要約(無し)のまま

    def test_full_summary_budget_is_independent_of_drain_timeout(self, monkeypatch):
        """全文要約の上限は stop の timeout(ドレイン)ではなく文字起こしの量に比例させる"""
        import live.session as session_module

        budgets: list[float] = []

        def fake_summarize_full(backend, texts, context, *, deadline, clock, cancel, on_skip=None):
            budgets.append(deadline - clock())
            time.sleep(0.3)  # ドレインの timeout(0.2秒)より長くかかる
            return SUMMARY_TEXT

        monkeypatch.setattr(session_module, "summarize_full", fake_summarize_full)
        session = make_session(FakeEngine(), backend=FakeBackend(), clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: session.backlog_sec == 0.0 and session._results)
        result = session.stop(timeout=0.2)
        assert result.final_summary == SUMMARY_TEXT
        assert budgets and budgets[0] > 60.0

    def test_summary_timeout_bounds_summary_and_sets_dedicated_cancel(self, monkeypatch, gates):
        """時間切れの全文要約には専用の cancel を立て、以降の LLM 呼び出しをさせない"""
        import live.session as session_module

        gate = gates()
        seen: list = []

        def fake_summarize_full(backend, texts, context, *, deadline, clock, cancel, on_skip=None):
            seen.append(cancel)
            gate.wait(timeout=5.0)
            return None

        monkeypatch.setattr(session_module, "summarize_full", fake_summarize_full)
        rec = Recorder()
        session = make_session(FakeEngine(), backend=FakeBackend(), rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: session.backlog_sec == 0.0 and session._results)
        began = time.monotonic()
        session.stop(timeout=WAIT_SEC, summary_timeout=0.3)
        assert time.monotonic() - began < 2.0
        assert seen and seen[0].is_set()
        assert any("全文からの最終要約が時間内に終わらなかった" in w for w in rec.warnings)

    def test_zero_summary_timeout_skips_llm_without_warning(self, gates):
        """エラー・取り消し経路(summary_timeout=0)は LLM を呼ばずにすぐ戻る"""
        rec = Recorder()
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: rec.summaries == [SUMMARY_TEXT])
        backend.gate = gates()  # 以降の呼び出しは返らない
        baseline = backend.calls
        began = time.monotonic()
        result = session.stop(timeout=0.2, summary_timeout=0.0)
        assert time.monotonic() - began < 0.5
        assert result.final_summary == SUMMARY_TEXT
        assert backend.calls == baseline
        assert not any("最終要約" in w for w in rec.warnings)

    def test_cancel_summary_ends_waiting_stop_quickly(self, gates):
        """要約待ちの stop は cancel_summary ですぐ戻り、ローリング要約への切り替えもしない"""
        rec = Recorder()
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: rec.summaries == [SUMMARY_TEXT])
        backend.gate = gates()
        baseline = backend.calls
        results: list = []
        stopper = threading.Thread(target=lambda: results.append(session.stop(timeout=WAIT_SEC)))
        stopper.start()
        assert wait_until(lambda: backend.calls == baseline + 1)  # 全文要約の呼び出し中
        began = time.monotonic()
        session.cancel_summary()
        session.cancel_summary()  # 冪等
        stopper.join(timeout=2.0)
        assert not stopper.is_alive()
        assert time.monotonic() - began < 0.5
        assert results[0].final_summary == SUMMARY_TEXT
        assert backend.calls == baseline + 1  # ローリング要約へのフォールバックは呼ばない

    def test_cancel_summary_before_stop(self):
        backend = FakeBackend()
        session = make_session(FakeEngine(), backend=backend, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: backend.calls >= 1)
        baseline = backend.calls
        session.cancel_summary()
        result = session.stop(timeout=WAIT_SEC)
        assert backend.calls == baseline
        assert result.final_summary == SUMMARY_TEXT

    def test_full_summary_deadline_counts_from_after_periodic_wait(self, monkeypatch):
        """周期要約の呼び出し中に停止しても、その待ちの分で全文要約の期限が削られない"""
        import live.full_summary as full_summary_module

        monkeypatch.setattr(full_summary_module, "FULL_SUMMARY_MARGIN_SEC", 0.1)
        rec = Recorder()

        class SlowFirst(FakeBackend):
            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    time.sleep(0.5)  # 停止時に周期要約が送信中
                return self.reply

        backend = SlowFirst()
        session = make_session(FakeEngine(), backend=backend, rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: backend.calls >= 1)
        result = session.stop(timeout=WAIT_SEC)
        assert backend.calls >= 2  # 全文要約が実行された
        assert result.final_summary == SUMMARY_TEXT
        assert not any("省き" in w for w in rec.warnings)

    def test_skipped_full_summary_is_warned_without_content(self, monkeypatch):
        """全文要約を省いたら on_warning で知らせる(塊数・秒数のみ、本文は出さない)"""
        import live.full_summary as full_summary_module

        monkeypatch.setattr(full_summary_module, "FULL_SUMMARY_MARGIN_SEC", -1.0)  # 必ず不足
        rec = Recorder()
        engine = FakeEngine(responses=["秘密の発話内容です"])
        session = make_session(engine, backend=FakeBackend(), rec=rec, clock=FakeClock(0.0))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: len(rec.segments) == 1)
        session.stop(timeout=WAIT_SEC)
        skipped = [w for w in rec.warnings if "全文" in w and "省き" in w]
        assert len(skipped) == 1, rec.warnings
        assert "1 塊" in skipped[0] and "秒" in skipped[0]
        assert "秘密" not in skipped[0]

    def test_wait_asr_idle_true_after_stop(self):
        session = make_session(FakeEngine())
        session.start()
        feed_audio(session, bursts(1))
        session.stop(timeout=WAIT_SEC)
        assert session.wait_asr_idle(timeout=WAIT_SEC) is True

    def test_wait_asr_idle_true_when_not_started(self):
        assert make_session(FakeEngine()).wait_asr_idle(timeout=0.0) is True

    def test_wait_asr_idle_false_while_transcribing(self, gates):
        gate = gates()
        session = make_session(FakeEngine(responses=[gate]))
        session.start()
        feed_audio(session, bursts(1))
        assert wait_until(lambda: session.backlog_sec > 0)
        session.stop(timeout=0.2)
        assert session.wait_asr_idle(timeout=0.1) is False
        gate.set()
        assert session.wait_asr_idle(timeout=WAIT_SEC) is True

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
