"""gui.workflow モジュールのユニットテスト (Tkinter・実スレッド非依存)。

`_timer_loop` は `time.sleep` を用いた無限ループのため、生スレッドでの実行は検証しない
(フレーキーの回避)。状態遷移・バリデーション・モード分岐を中心にテストする。
"""

from pathlib import Path

import pytest

import gui.workflow as workflow_module
from config import VoiceNoteConfig
from gui.constants import MODE_RECORD_ONLY, MODE_RECORD_TRANSCRIBE
from gui.workflow import (
    RecordingWorkflow,
    WorkflowCallbacks,
    validate_start,
    validate_transcribe_only,
)


class ImmediateThread:
    """threading.Thread の代わりにターゲットを同期的に実行するスタブ。"""

    def __init__(self, target, args=(), daemon=None):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


class DeferredThread:
    """スレッドを起動せずターゲットをキャプチャするだけのスタブ (タイマーループ用)。"""

    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        pass


class FakeRecorder:
    def __init__(self, device_id=None, fail_start=False, fail_get_data=False, data=None):
        self.device_id = device_id
        self._fail_start = fail_start
        self._fail_get_data = fail_get_data
        self._data = data if data is not None else [0.0] * 16000
        self.started = False
        self.stopped = False

    def start(self):
        if self._fail_start:
            raise RuntimeError("デバイスが使用できません")
        self.started = True

    def stop(self):
        self.stopped = True

    def get_data(self):
        if self._fail_get_data:
            raise RuntimeError("録音データがありません")
        return self._data


class SpyCallbacks:
    def __init__(self):
        self.status: list[str] = []
        self.logs: list[str] = []
        self.recording_started = 0
        self.processing_started = 0
        self.done: list[Path] = []
        self.record_only_done: list[Path] = []
        self.errors: list[str] = []

    def build(self) -> WorkflowCallbacks:
        return WorkflowCallbacks(
            on_status=self.status.append,
            on_log=self.logs.append,
            on_recording_started=lambda: setattr(self, "recording_started", self.recording_started + 1),
            on_processing_started=lambda: setattr(self, "processing_started", self.processing_started + 1),
            on_done=self.done.append,
            on_record_only_done=self.record_only_done.append,
            on_error=self.errors.append,
        )


class TestValidateStart:
    def test_requires_save_folder_for_record_transcribe(self):
        error = validate_start(MODE_RECORD_TRANSCRIBE, "")
        assert error is not None
        assert error[0] == "設定が必要"

    def test_allows_empty_save_folder_for_record_only(self):
        assert validate_start(MODE_RECORD_ONLY, "") is None

    def test_passes_when_save_folder_set(self):
        assert validate_start(MODE_RECORD_TRANSCRIBE, "/tmp/notes") is None


class TestValidateTranscribeOnly:
    def test_requires_audio_path(self):
        error = validate_transcribe_only("", "/tmp/notes")
        assert error == ("ファイル未選択", "音声ファイルを選択してください")

    def test_requires_save_folder(self):
        error = validate_transcribe_only("a.wav", "")
        assert error is not None
        assert error[0] == "設定が必要"

    def test_passes_when_both_set(self):
        assert validate_transcribe_only("a.wav", "/tmp/notes") is None


class TestRecordingWorkflowStart:
    def test_start_success_sets_recording_state(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        spy = SpyCallbacks()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda device_id: FakeRecorder(device_id),
        )

        error = wf.start(device_id=1, device_label="[1] マイク")

        assert error is None
        assert wf.is_recording is True
        assert spy.recording_started == 1
        assert any("録音開始" in msg for msg in spy.logs)

    def test_start_failure_returns_error_and_stays_idle(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        spy = SpyCallbacks()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda device_id: FakeRecorder(device_id, fail_start=True),
        )

        error = wf.start(device_id=None, device_label="デバイスなし")

        assert error == "デバイスが使用できません"
        assert wf.is_recording is False
        assert spy.recording_started == 0


class TestRecordingWorkflowStopAndProcess:
    def test_record_only_mode_skips_transcription(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved = tmp_path / "out.wav"
        monkeypatch.setattr(workflow_module, "save_wav", lambda data, dest: saved)

        def fail_transcribe(*args, **kwargs):
            raise AssertionError("RECORD_ONLY では transcribe_and_save が呼ばれてはならない")

        monkeypatch.setattr(workflow_module, "transcribe_and_save", fail_transcribe)

        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), recorder_factory=lambda d: recorder)

        # start() が起動するタイマースレッドは無限ループのため実行させない
        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        wf.start(device_id=None, device_label="デバイスなし")

        # stop_and_process が起動する処理スレッドは同期実行させて結果を検証する
        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        wf.stop_and_process(tmp_path, MODE_RECORD_ONLY)

        assert wf.is_recording is False
        assert recorder.stopped is True
        assert spy.record_only_done == [saved]
        assert spy.done == []

    def test_record_transcribe_mode_runs_transcription(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved_wav = tmp_path / "out.wav"
        saved_note = tmp_path / "out.md"
        monkeypatch.setattr(workflow_module, "save_wav", lambda data, dest: saved_wav)
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: saved_note,
        )

        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), recorder_factory=lambda d: recorder)

        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        wf.start(device_id=None, device_label="デバイスなし")

        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        wf.stop_and_process(tmp_path, MODE_RECORD_TRANSCRIBE)

        assert spy.done == [saved_note]
        assert spy.record_only_done == []

    def test_save_wav_failure_reports_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        def fail_save_wav(data, dest):
            raise OSError("ディスク書き込みエラー")

        monkeypatch.setattr(workflow_module, "save_wav", fail_save_wav)

        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), recorder_factory=lambda d: recorder)

        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        wf.start(device_id=None, device_label="デバイスなし")

        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        wf.stop_and_process(tmp_path, MODE_RECORD_TRANSCRIBE)

        assert spy.done == []
        assert spy.record_only_done == []
        assert any("エラー" in msg for msg in spy.errors)

    def test_unexpected_exception_in_process_audio_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # _process_audio の内側 try (save_wav 呼び出し) より前段で例外を起こし、
        # 外側の except (「予期せぬエラー」ハンドラ) を通ることを検証する
        errors: list[str] = []
        log_calls = {"count": 0}

        def raise_on_process_audio_log(msg: str):
            # stop_and_process 自体が発行する3回の on_log (停止/クローズ/取得完了) の後、
            # _process_audio の最初の on_log ("WAVファイルを書き込み中...") で例外を発生させる
            log_calls["count"] += 1
            if log_calls["count"] > 3:
                raise RuntimeError("ログ書き込み中の想定外エラー")

        callbacks = WorkflowCallbacks(
            on_status=lambda msg: None,
            on_log=raise_on_process_audio_log,
            on_recording_started=lambda: None,
            on_processing_started=lambda: None,
            on_done=lambda path: None,
            on_record_only_done=lambda path: None,
            on_error=errors.append,
        )

        recorder = FakeRecorder()
        log_file = tmp_path / "app.log"
        wf = RecordingWorkflow(
            VoiceNoteConfig(), callbacks, recorder_factory=lambda d: recorder, log_file=log_file
        )

        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        wf.start(device_id=None, device_label="デバイスなし")
        log_calls["count"] = 0  # start() 内の on_log 呼び出し分をリセット

        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        wf.stop_and_process(tmp_path, MODE_RECORD_TRANSCRIBE)

        assert len(errors) == 1
        assert log_file.name in errors[0]

    def test_get_data_failure_reports_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        spy = SpyCallbacks()
        recorder = FakeRecorder(fail_get_data=True)
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), recorder_factory=lambda d: recorder)

        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        wf.start(device_id=None, device_label="デバイスなし")

        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        wf.stop_and_process(tmp_path, MODE_RECORD_TRANSCRIBE)

        assert spy.errors
        assert spy.done == []
        assert spy.record_only_done == []


class TestRecordingWorkflowTranscribeOnly:
    def test_runs_transcription_and_reports_done(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)
        saved_note = tmp_path / "out.md"
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: saved_note,
        )

        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build())

        wf.run_transcribe_only(tmp_path / "in.wav")

        assert spy.processing_started == 1
        assert spy.done == [saved_note]

    def test_transcription_error_is_reported(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(workflow_module.threading, "Thread", ImmediateThread)

        def fail(*args, **kwargs):
            raise RuntimeError("文字起こし失敗")

        monkeypatch.setattr(workflow_module, "transcribe_and_save", fail)

        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build())

        wf.run_transcribe_only(tmp_path / "in.wav")

        assert spy.done == []
        assert any("文字起こしエラー" in msg for msg in spy.errors)


class TestRecordingWorkflowShutdown:
    def test_shutdown_stops_recorder_and_clears_state(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(workflow_module.threading, "Thread", DeferredThread)
        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), recorder_factory=lambda d: recorder)
        wf.start(device_id=None, device_label="デバイスなし")

        wf.shutdown()

        assert wf.is_recording is False
        assert recorder.stopped is True

    def test_shutdown_when_not_recording_is_noop(self):
        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build())

        wf.shutdown()  # 例外を送出しないこと

        assert wf.is_recording is False
