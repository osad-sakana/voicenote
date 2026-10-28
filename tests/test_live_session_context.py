"""LiveSession の参考情報（参加者・メモ）対応のテスト。"""

import threading

import numpy as np
import pytest

from live.context import LiveContext
from live.session import LiveSession, LiveSettings
from live.vad import EnergyVAD

SR = 16000
CHUNK = 1600
WAIT_SEC = 5.0
SUMMARY_TEXT = "現在の話題: テスト\n進捗:\n- 要点その1が話された"


class Text:
    def __init__(self, text):
        self.text = text
        self.no_speech_prob = None
        self.avg_logprob = None


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def load(self):
        return None

    def transcribe(self, samples, language):
        self.calls += 1
        return Text(f"発話{self.calls}")


class RecordingBackend:
    def __init__(self):
        self.messages: list[list[dict]] = []
        self._lock = threading.Lock()

    def complete(self, messages):
        with self._lock:
            self.messages.append(list(messages))
        return SUMMARY_TEXT

    def list_models(self):
        return ["m"]


def bursts(n: int) -> np.ndarray:
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(SR, dtype=np.float32)
    return np.concatenate([np.concatenate([tone, silence]) for _ in range(n)])


def feed_audio(session: LiveSession, audio: np.ndarray) -> None:
    for start in range(0, len(audio), CHUNK):
        session.feed(audio[start : start + CHUNK])


def make_session(backend, **kwargs) -> LiveSession:
    settings = LiveSettings(summary_poll_sec=0.01, summary_interval_sec=3600)
    return LiveSession(
        settings, engine=FakeEngine(), vad=EnergyVAD(), backend=backend, clock=lambda: 0.0, **kwargs
    )


def _user(messages) -> str:
    return messages[1]["content"]


class TestLiveSessionContext:
    def test_initial_context_reaches_summary(self):
        backend = RecordingBackend()
        session = make_session(backend, context=LiveContext(participants=("田中",)))
        session.start()
        feed_audio(session, bursts(1))
        session.stop(timeout=WAIT_SEC)
        assert backend.messages
        assert "- 田中" in _user(backend.messages[-1])

    def test_default_has_no_context(self):
        backend = RecordingBackend()
        session = make_session(backend)
        session.start()
        feed_audio(session, bursts(1))
        session.stop(timeout=WAIT_SEC)
        assert "参考情報" not in _user(backend.messages[-1])

    def test_set_context_applies_to_final_summary(self):
        backend = RecordingBackend()
        session = make_session(backend)
        session.start()
        # 周期要約・最終要約のどちらが先に走っても、設定後の要約には反映される
        session.set_context(LiveContext(memo="議題: 予算"))
        feed_audio(session, bursts(1))
        session.stop(timeout=WAIT_SEC)
        assert "議題: 予算" in _user(backend.messages[-1])

    def test_set_context_without_backend_is_noop(self):
        session = make_session(None)
        session.set_context(LiveContext(memo="x"))
        session.start()
        result = session.stop(timeout=WAIT_SEC)
        assert result.final_summary == ""

    @pytest.mark.parametrize("backend", [RecordingBackend(), None])
    def test_set_context_rejects_wrong_type(self, backend):
        session = make_session(backend)
        with pytest.raises(TypeError):
            session.set_context("メモ")

    def test_concurrent_set_context_while_running(self):
        backend = RecordingBackend()
        session = make_session(backend)
        session.start()
        errors: list[BaseException] = []
        contexts = [LiveContext(participants=(f"人{i}",)) for i in range(8)]

        def setter(ctx):
            try:
                for _ in range(200):
                    session.set_context(ctx)
            except BaseException as e:  # pragma: no cover - 失敗時のみ
                errors.append(e)

        threads = [threading.Thread(target=setter, args=(c,)) for c in contexts]
        for t in threads:
            t.start()
        feed_audio(session, bursts(2))
        for t in threads:
            t.join()
        result = session.stop(timeout=WAIT_SEC)
        assert errors == []
        assert result.complete is True
        assert backend.messages
        expected = {f"- 人{i}" for i in range(8)}
        for messages in backend.messages:
            lines = [line for line in _user(messages).splitlines() if line.startswith("- 人")]
            assert len(lines) <= 1
            assert set(lines) <= expected
