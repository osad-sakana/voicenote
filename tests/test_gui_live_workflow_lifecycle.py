"""gui/live_workflow.py のテスト: 排他・表示とメーター・shutdown・設定差し替え

Fake 群と Harness は tests/live_workflow_fakes.py にある。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import VoiceNoteConfig
from gui.live_state import LivePhase
from gui.live_workflow import BANNER_LOW_INPUT
from live.session_models import LiveSegment
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


class TestFailedPrepareDiscardsLateCallbacks:
    """準備の失敗・取り消し後に、解放しきれなかったセッションから遅れて届く通知を捨てる"""

    def _failed(self, tmp_path) -> Harness:
        h = Harness(tmp_path, fail_start=True)  # セッション起動後に録音開始で失敗
        h.start()
        assert h.spy.errors
        assert h.state.phase is LivePhase.IDLE
        assert len(h.sessions) == 1
        return h

    def test_late_status_and_warning_are_ignored(self, tmp_path):
        h = self._failed(tmp_path)
        before, notified = h.state, len(h.spy.states)
        callbacks = h.sessions[0].callbacks
        callbacks.on_status("遅れた状態")
        callbacks.on_warning("遅れた警告")
        callbacks.on_segment(0, 0.0, 1.0, "遅れた区間")
        assert h.state is before
        assert len(h.spy.states) == notified

    def test_late_callbacks_do_not_touch_next_run(self, tmp_path):
        h = self._failed(tmp_path)
        h.fail_start = False
        assert h.start() is None
        before = h.state
        h.sessions[0].callbacks.on_warning("前回の実行の警告")
        assert h.state is before
        assert "前回の実行の警告" not in h.state.warnings


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
