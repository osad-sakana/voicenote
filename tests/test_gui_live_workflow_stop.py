"""gui/live_workflow.py のテスト: 停止(WAV確定 → ストリーム停止 → ドレイン)・放棄ストリーム・ドレイン中断

Fake 群と Harness は tests/live_workflow_fakes.py にある。
"""

from __future__ import annotations

import pytest

from config import VoiceNoteConfig
from gui.live_state import LivePhase
from live.session_models import LiveResult
from tests.live_workflow_fakes import (
    DeferredThread,
    Harness,
    ImmediateThread,
    reset_deferred_threads,
    run_pending,
)


@pytest.fixture(autouse=True)
def _reset_deferred():
    reset_deferred_threads()
    yield
    reset_deferred_threads()


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
        note = h.saved_notes[0][1]
        assert "OpenAI" in note
        # CLI 向けの文言（--config / この実行は）はノートにも出さない
        assert "--config" not in note
        assert "この実行は" not in note

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
        def failing_saver(folder, body, audio_file=None, participants=()):
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
