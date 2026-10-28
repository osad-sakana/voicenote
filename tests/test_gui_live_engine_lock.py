"""SerializedEngine のロック取得タイムアウトのテスト

放棄した asr スレッドが transcribe で固まりロックを持ち続けても、次の実行の load() が
無通知で待ち続けないこと(タイムアウトで失敗扱いにし、セッションが on_warning で知らせる)。
"""

from __future__ import annotations

import threading

import pytest

from gui.live_workflow_support import (
    ENGINE_LOCK_TIMEOUT_SEC,
    EngineBusyError,
    SerializedEngine,
)
from tests.live_workflow_fakes import Harness


class _Inner:
    def __init__(self):
        self.loaded = 0
        self.transcribed = 0
        self.prompts: list[str | None] = []

    def load(self):
        self.loaded += 1

    def transcribe(self, samples, language, prompt=None):
        self.transcribed += 1
        self.prompts.append(prompt)
        return "text"


class TestSerializedEngineLoad:
    def test_default_timeout_constant(self):
        assert ENGINE_LOCK_TIMEOUT_SEC == 30.0

    def test_load_when_lock_free(self):
        inner, lock = _Inner(), threading.Lock()
        SerializedEngine(inner, lock, load_timeout=0.01).load()
        assert inner.loaded == 1
        assert not lock.locked()  # 解放されている

    def test_load_fails_when_previous_run_holds_lock(self):
        inner, lock = _Inner(), threading.Lock()
        lock.acquire()  # 放棄した asr スレッドが transcribe 中
        try:
            with pytest.raises(EngineBusyError, match="前回の処理が終わっていません"):
                SerializedEngine(inner, lock, load_timeout=0.01).load()
        finally:
            lock.release()
        assert inner.loaded == 0

    def test_lock_released_when_inner_load_raises(self):
        class _Broken(_Inner):
            def load(self):
                raise RuntimeError("boom")

        lock = threading.Lock()
        with pytest.raises(RuntimeError):
            SerializedEngine(_Broken(), lock, load_timeout=0.01).load()
        assert not lock.locked()

    def test_transcribe_still_serialized(self):
        inner, lock = _Inner(), threading.Lock()
        assert SerializedEngine(inner, lock).transcribe(None, "ja") == "text"
        assert inner.transcribed == 1
        assert not lock.locked()

    def test_transcribe_forwards_prompt(self):
        inner, lock = _Inner(), threading.Lock()
        engine = SerializedEngine(inner, lock)
        engine.transcribe(None, "ja", prompt="参加者: 田中。")
        engine.transcribe(None, "ja")
        assert inner.prompts == ["参加者: 田中。", None]


class TestWorkflowPassesTimeout:
    def test_session_engine_uses_injected_timeout(self, tmp_path):
        h = Harness(tmp_path, engine_lock_timeout=0.01)
        assert h.start() is None
        engine = h.sessions[0].engine
        # 前回の実行の asr スレッドがロックを保持している状態
        with h.wf._engine_lock, pytest.raises(EngineBusyError):
            engine.load()


class TestSessionWarnsWhenEngineBusy:
    def test_load_timeout_warns_and_session_keeps_running(self):
        """実 LiveSession: ロックが取れなければ on_warning で知らせ、stop は正常に返る"""
        from tests.test_live_session import (
            WAIT_SEC,
            FakeEngine,
            Recorder,
            bursts,
            feed_audio,
            make_session,
        )

        rec, inner, lock = Recorder(), FakeEngine(), threading.Lock()
        lock.acquire()  # 前回の実行の asr スレッドが transcribe で固まっている
        try:
            session = make_session(SerializedEngine(inner, lock, load_timeout=0.01), rec=rec)
            session.start()
            feed_audio(session, bursts(1))
            result = session.stop(timeout=WAIT_SEC)
        finally:
            lock.release()
        assert result.segments == ()
        assert inner.calls == 0 and inner.load_thread is None
        assert any("前回の処理が終わっていません" in w for w in rec.warnings)
