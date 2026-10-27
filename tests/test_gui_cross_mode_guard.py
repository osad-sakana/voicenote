"""ライブと通常録音の「放棄したストリームがまだ閉じている」ガードが互いに効くことのテスト

stop_with_timeout() がタイムアウトした経路(macOS PortAudio のハング)では、放棄した
ストリームが解放されるまで、相手側のモードの録音開始も拒否する(猶予を過ぎたら許可する)。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from config import VoiceNoteConfig
from gui.constants import MODE_RECORD_TRANSCRIBE
from gui.live_state import LivePhase
from gui.live_workflow import validate_live_start
from gui.mode_logic import other_mode_rejection
from gui.workflow import RecordingWorkflow
from tests.live_workflow_fakes import Harness, reset_deferred_threads
from tests.test_workflow import DeferredThread as RecDeferredThread
from tests.test_workflow import FakeRecorder as RecFakeRecorder
from tests.test_workflow import ImmediateThread as RecImmediateThread
from tests.test_workflow import SpyCallbacks as RecSpyCallbacks


@pytest.fixture(autouse=True)
def _reset_deferred():
    reset_deferred_threads()
    yield
    reset_deferred_threads()


@pytest.fixture(autouse=True)
def _no_transcription(monkeypatch, tmp_path):
    import gui.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "transcribe_and_save",
        lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
    )


def _timed_out_recording(tmp_path, *, closing=True) -> tuple[RecordingWorkflow, RecFakeRecorder]:
    """通常録音を開始 → 停止し、ストリーム停止がタイムアウトした状態を作る"""
    recorder = RecFakeRecorder(stop_timeout_result=False)
    recorder.is_closing = lambda: closing
    wf = RecordingWorkflow(
        VoiceNoteConfig(),
        RecSpyCallbacks().build(),
        recorder_factory=lambda d, dest: recorder,
        stop_timeout=0.05,
        thread_factory=RecDeferredThread,  # タイマーループを起動しない
    )
    wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)
    wf._thread_factory = RecImmediateThread
    wf.stop_and_process(MODE_RECORD_TRANSCRIBE)
    return wf, recorder


def _timed_out_live(tmp_path) -> Harness:
    """ライブを開始 → 停止し、ストリーム停止がタイムアウトした状態を作る"""
    h = Harness(tmp_path, stop_result=False)
    assert h.start() is None
    h.wf.stop()
    h.recorders[0].closing = True
    assert h.state.phase is LivePhase.IDLE
    return h


# ──────────────── RecordingWorkflow.has_pending_close ────────────────


class TestRecordingHasPendingClose:
    def test_false_without_timeout(self, tmp_path):
        wf = RecordingWorkflow(VoiceNoteConfig(), RecSpyCallbacks().build())
        assert wf.has_pending_close() is False

    def test_true_while_abandoned_stream_is_closing(self, tmp_path):
        wf, _ = _timed_out_recording(tmp_path)
        assert wf.has_pending_close() is True

    def test_false_once_stream_closed(self, tmp_path):
        wf, recorder = _timed_out_recording(tmp_path)
        recorder.is_closing = lambda: False
        assert wf.has_pending_close() is False

    def test_false_after_grace_period(self, tmp_path):
        wf, _ = _timed_out_recording(tmp_path)
        wf._pending_close_since = time.monotonic() - 10.0
        assert wf.has_pending_close() is False

    def test_query_does_not_clear_own_guard(self, tmp_path):
        wf, _ = _timed_out_recording(tmp_path)
        wf.has_pending_close()
        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)
        assert error is not None and "解放" in error


# ──────────────── LiveWorkflow.has_pending_close ────────────────


class TestLiveHasPendingClose:
    def test_false_without_timeout(self, tmp_path):
        h = Harness(tmp_path)
        h.start()
        h.wf.stop()
        assert h.wf.has_pending_close() is False

    def test_true_while_abandoned_stream_is_closing(self, tmp_path):
        h = _timed_out_live(tmp_path)
        assert h.wf.is_active is False  # 従来の is_active では見えない
        assert h.wf.has_pending_close() is True

    def test_false_once_stream_closed(self, tmp_path):
        h = _timed_out_live(tmp_path)
        h.recorders[0].closing = False
        assert h.wf.has_pending_close() is False

    def test_false_after_grace_period(self, tmp_path):
        h = _timed_out_live(tmp_path)
        h.clock_now += 1000.0
        assert h.wf.has_pending_close() is False

    def test_query_does_not_clear_own_guard(self, tmp_path):
        h = _timed_out_live(tmp_path)
        h.wf.has_pending_close()
        error = h.start()
        assert error is not None and "解放" in error


# ──────────────── 相互のガード ────────────────


class TestLiveTimeoutBlocksNormalRecording:
    def test_normal_recording_rejected(self, tmp_path):
        h = _timed_out_live(tmp_path)
        rejection = other_mode_rejection(
            live_active=h.wf.is_active, live_stream_closing=h.wf.has_pending_close()
        )
        assert rejection is not None
        assert "解放" in rejection[1]

    def test_allowed_after_grace_period(self, tmp_path):
        h = _timed_out_live(tmp_path)
        h.clock_now += 1000.0
        rejection = other_mode_rejection(
            live_active=h.wf.is_active, live_stream_closing=h.wf.has_pending_close()
        )
        assert rejection is None


class TestNormalTimeoutBlocksLive:
    def test_live_start_rejected_by_can_start(self, tmp_path):
        rec, _ = _timed_out_recording(tmp_path)
        h = Harness(tmp_path, can_start=lambda: not rec.has_pending_close())
        assert h.start() is not None
        assert h.state.phase is LivePhase.IDLE
        assert h.recorders == []

    def test_live_start_rejected_by_validation(self, tmp_path):
        rec, _ = _timed_out_recording(tmp_path)
        config = VoiceNoteConfig(save_folder=str(tmp_path))
        error = validate_live_start(
            config, "[1] マイク", mic_busy=False, stream_closing=rec.has_pending_close()
        )
        assert error is not None
        assert "解放" in error[1]

    def test_live_start_allowed_after_grace_period(self, tmp_path):
        rec, _ = _timed_out_recording(tmp_path)
        rec._pending_close_since = time.monotonic() - 10.0
        h = Harness(tmp_path, can_start=lambda: not rec.has_pending_close())
        assert h.start() is None
        assert h.state.phase is LivePhase.RECORDING


# ──────────────── App / LiveController の配線 ────────────────


def _import_or_skip(module: str):
    try:
        return __import__(module, fromlist=["_"])
    except Exception as e:  # pragma: no cover - GUI 依存が無い環境
        pytest.skip(f"GUI 依存を読み込めません: {e}")


class TestControllerWiring:
    def _controller(self, *, busy=False, recording=False, rec_closing=False, live_closing=False):
        live_controller = _import_or_skip("gui.live_controller")
        app = SimpleNamespace(
            _busy=busy,
            _workflow=SimpleNamespace(
                is_recording=recording, has_pending_close=lambda: rec_closing
            ),
        )
        stub = SimpleNamespace(
            _app=app,
            _workflow=SimpleNamespace(has_pending_close=lambda: live_closing),
        )
        return live_controller.LiveController, stub

    def test_can_start_false_while_normal_stream_closing(self):
        cls, stub = self._controller(rec_closing=True)
        assert cls._can_start(stub) is False

    def test_can_start_true_when_idle(self):
        cls, stub = self._controller()
        assert cls._can_start(stub) is True

    def test_can_start_keeps_existing_checks(self):
        cls, stub = self._controller(recording=True)
        assert cls._can_start(stub) is False

    def test_has_pending_close_passthrough(self):
        cls, stub = self._controller(live_closing=True)
        assert cls.has_pending_close(stub) is True


class TestAppRejectWiring:
    def _call(self, monkeypatch, *, live_active=False, live_closing=False):
        app_module = _import_or_skip("gui.app")
        shown: list[tuple[str, str]] = []
        monkeypatch.setattr(app_module.messagebox, "showwarning", lambda *a: shown.append(a))
        stub = SimpleNamespace(
            _live=SimpleNamespace(is_active=live_active, has_pending_close=lambda: live_closing)
        )
        return app_module.App._reject_if_live_active(stub), shown

    def test_rejects_while_live_stream_closing(self, monkeypatch):
        rejected, shown = self._call(monkeypatch, live_closing=True)
        assert rejected is True
        assert shown and "解放" in shown[0][1]

    def test_allows_when_live_idle(self, monkeypatch):
        rejected, shown = self._call(monkeypatch)
        assert rejected is False
        assert shown == []
