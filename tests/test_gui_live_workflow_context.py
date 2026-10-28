"""gui/live_workflow.py のテスト: 参考情報（参加者・メモ）の保持・セッションへの反映・ノート保存

Fake 群と Harness は tests/live_workflow_fakes.py にある。
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from gui.live_state import LivePhase
from live.context import LiveContext
from tests.live_workflow_fakes import (
    DeferredThread,
    Harness,
    ImmediateThread,
    reset_deferred_threads,
    run_pending,
)

SECRET_MEMO = "極秘メモ: 買収案件"
SECRET_NAME = "秘密太郎"
CTX = LiveContext(participants=(SECRET_NAME, "佐藤"), memo=SECRET_MEMO)


@pytest.fixture(autouse=True)
def _reset_deferred():
    reset_deferred_threads()
    yield
    reset_deferred_threads()


def _assert_note_has_context(h: Harness, ctx: LiveContext = CTX, index: int = 0) -> None:
    body = h.saved_notes[index][1]
    assert "## 参加者" in body
    for name in ctx.participants:
        assert f"- {name}" in body
    assert f"## メモ\n\n{ctx.memo}" in body
    assert h.saved_participants[index] == ctx.participants


def _draining(h: Harness) -> None:
    h.wf._thread_factory = DeferredThread
    h.wf.stop()
    DeferredThread.instances[0].run()  # 停止ワーカー(WAV確定)だけ
    assert h.state.phase is LivePhase.DRAINING
    h.wf._thread_factory = ImmediateThread


class TestUpdateContextBasics:
    def test_default_is_empty(self, tmp_path):
        h = Harness(tmp_path)
        assert h.wf.context == LiveContext()

    def test_none_resets_to_empty(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.wf.update_context(None)
        assert h.wf.context == LiveContext()

    def test_rejects_wrong_type(self, tmp_path):
        h = Harness(tmp_path)
        with pytest.raises(TypeError):
            h.wf.update_context("メモ")

    def test_without_context_note_and_saver_are_legacy(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.wf.stop()
        body = h.saved_notes[0][1]
        assert "## 参加者" not in body and "## メモ" not in body
        assert h.saved_participants == [()]
        assert h.sessions[0].context == LiveContext()


class TestContextBeforeAndDuringRun:
    def test_context_set_while_idle_is_used_at_start(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        assert h.sessions[0].context == CTX
        assert h.sessions[0].context_updates == []
        h.wf.stop()
        _assert_note_has_context(h)

    def test_context_set_while_preparing_before_session(self, tmp_path):
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()
        assert h.state.phase is LivePhase.PREPARING
        h.wf.update_context(CTX)
        run_pending()
        assert h.sessions[0].context == CTX

    def test_context_updated_during_session_creation_is_not_lost(self, tmp_path):
        """セッション生成中(ロック外)に更新されても、生成後に最新の値を反映する"""
        h = Harness(tmp_path, thread_factory=DeferredThread)
        h.start()
        original = h._make_session

        def make_and_update(*args, **kwargs):
            session = original(*args, **kwargs)
            h.wf.update_context(CTX)
            return session

        h.wf._deps = replace(h.wf._deps, session_factory=make_and_update)
        run_pending()
        assert h.sessions[0].context == LiveContext()
        assert h.sessions[0].context_updates[-1] == CTX

    def test_context_set_while_recording_is_pushed_to_session(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        assert h.state.phase is LivePhase.RECORDING
        h.wf.update_context(CTX)
        assert h.sessions[0].context_updates == [CTX]
        h.wf.stop()
        _assert_note_has_context(h)

    def test_latest_context_is_used_for_note(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(LiveContext(memo="古いメモ"))
        h.start()
        h.wf.update_context(CTX)
        h.wf.stop()
        assert "古いメモ" not in h.saved_notes[0][1]
        _assert_note_has_context(h)

    def test_context_is_kept_for_next_run(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        h.wf.stop()
        assert h.state.phase is LivePhase.IDLE
        assert h.wf.context == CTX
        h.start()
        assert h.sessions[1].context == CTX

    def test_session_set_context_failure_is_swallowed_without_content(self, tmp_path, caplog):
        h = Harness(tmp_path)
        h.start()

        def broken(ctx):
            raise RuntimeError("boom")

        h.sessions[0].set_context = broken
        with caplog.at_level(logging.DEBUG, logger="voicenote"):
            h.wf.update_context(CTX)
        assert h.wf.context == CTX
        assert SECRET_MEMO not in caplog.text and SECRET_NAME not in caplog.text
        h.wf.stop()
        _assert_note_has_context(h)


class TestContextInAllSavePaths:
    def test_stop_order_is_unchanged_with_context(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
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
        _assert_note_has_context(h)

    def test_context_updated_while_draining_is_in_note(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        _draining(h)
        h.wf.update_context(CTX)
        run_pending()
        _assert_note_has_context(h)

    def test_abort_drain_partial_note_has_context(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        _draining(h)
        assert h.wf.abort_drain() is True
        _assert_note_has_context(h)
        assert "未処理あり: はい" in h.saved_notes[0][1]

    def test_session_stop_failure_partial_note_has_context(self, tmp_path):
        h = Harness(tmp_path, session_fail_stop=True)
        h.wf.update_context(CTX)
        h.start()
        h.wf.stop()
        _assert_note_has_context(h)

    def test_shutdown_while_recording_partial_note_has_context(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        h.wf.shutdown()
        _assert_note_has_context(h)

    def test_shutdown_while_draining_partial_note_has_context(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        _draining(h)
        h.wf.shutdown()
        _assert_note_has_context(h)

    def test_shutdown_while_stream_stop_hangs_note_has_context(self, tmp_path):
        h = Harness(tmp_path)
        h.wf.update_context(CTX)
        h.start()
        h.recorders[0].on_stop = h.wf.shutdown
        h.wf.stop()
        assert len(h.saved_notes) == 1
        _assert_note_has_context(h)


class TestContextPrivacy:
    def test_logs_and_callbacks_do_not_contain_context(self, tmp_path, caplog):
        h = Harness(tmp_path)
        with caplog.at_level(logging.DEBUG, logger="voicenote"):
            h.wf.update_context(CTX)
            h.start()
            h.wf.update_context(CTX)
            h.wf.stop()
        texts = [caplog.text, *h.spy.logs, *h.spy.errors]
        for text in texts:
            assert SECRET_MEMO not in text and SECRET_NAME not in text
