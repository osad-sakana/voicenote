"""gui.workflow モジュールのユニットテスト (Tkinter・実スレッド非依存)。

`_timer_loop` は `time.sleep` を用いた無限ループのため、生スレッドでの実行は検証しない
(フレーキーの回避)。状態遷移・バリデーション・モード分岐を中心にテストする。

スレッド生成は `RecordingWorkflow` に注入する `thread_factory` で差し替える
(グローバルな `threading.Thread` を monkeypatch しない)。 `wf._thread_factory` を
テスト内で直接差し替えることで、`start()` はタイマーループを起動させず、
`stop_and_process()`/`shutdown()` だけ同期実行・実スレッド実行を選べる。
"""

import threading
import time
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


class CapturingThread(threading.Thread):
    """実スレッドとして動きつつ、生成したインスタンスをテスト側で回収できるスタブ。

    テスト終了時に確実に join してリークを防ぐために使う。
    """

    def __init__(self, target, args=(), daemon=None):
        super().__init__(target=target, args=args, daemon=daemon)
        self.instances.append(self)

    instances: list["CapturingThread"] = []


class DeferredThread:
    """スレッドを起動せずターゲットをキャプチャするだけのスタブ (タイマーループ用)。"""

    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        pass


class FakeRecorder:
    def __init__(
        self,
        device_id=None,
        rec_dest=None,
        fail_start=False,
        fail_finalize=False,
        finalized_path=None,
        stop_timeout_result=True,
        block_on_stop: threading.Event | None = None,
    ):
        self.device_id = device_id
        self.rec_dest = rec_dest
        self._fail_start = fail_start
        self._fail_finalize = fail_finalize
        self._finalized_path = finalized_path if finalized_path is not None else Path("out.wav")
        self._stop_timeout_result = stop_timeout_result
        self._block_on_stop = block_on_stop
        self.started = False
        self.stopped = False
        self.stop_timeout_used = None
        self.finalize_called = False

    def start(self):
        if self._fail_start:
            raise RuntimeError("デバイスが使用できません")
        self.started = True

    def stop(self):
        self.stopped = True

    def stop_with_timeout(self, timeout):
        self.stop_timeout_used = timeout
        if self._block_on_stop is not None:
            self._block_on_stop.wait()
        self.stopped = True
        return self._stop_timeout_result

    def is_closing(self):
        return False

    def finalize_recording(self, timeout=2.0):
        self.finalize_called = True
        if self._fail_finalize:
            return None
        return self._finalized_path


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
            on_recording_started=lambda: setattr(
                self, "recording_started", self.recording_started + 1
            ),
            on_processing_started=lambda: setattr(
                self, "processing_started", self.processing_started + 1
            ),
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
    def test_start_success_sets_recording_state(self, tmp_path: Path):
        spy = SpyCallbacks()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda device_id, rec_dest: FakeRecorder(device_id, rec_dest),
            thread_factory=DeferredThread,
        )

        error = wf.start(device_id=1, device_label="[1] マイク", rec_dest=tmp_path)

        assert error is None
        assert wf.is_recording is True
        assert spy.recording_started == 1
        assert any("録音開始" in msg for msg in spy.logs)

    def test_start_failure_returns_error_and_stays_idle(self, tmp_path: Path):
        spy = SpyCallbacks()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda device_id, rec_dest: FakeRecorder(
                device_id, rec_dest, fail_start=True
            ),
            thread_factory=DeferredThread,
        )

        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        assert error == "デバイスが使用できません"
        assert wf.is_recording is False
        assert spy.recording_started == 0


class TestRecordingWorkflowStopAndProcess:
    def test_record_only_mode_skips_transcription(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved = tmp_path / "out.wav"

        def fail_transcribe(*args, **kwargs):
            raise AssertionError("RECORD_ONLY では transcribe_and_save が呼ばれてはならない")

        monkeypatch.setattr(workflow_module, "transcribe_and_save", fail_transcribe)

        spy = SpyCallbacks()
        recorder = FakeRecorder(finalized_path=saved)
        # start() が起動するタイマースレッドは無限ループのため実行させない
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        # stop_and_process が起動する処理スレッドは同期実行させて結果を検証する
        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_ONLY)

        assert wf.is_recording is False
        assert recorder.finalize_called is True
        assert recorder.stopped is True
        assert spy.record_only_done == [saved]
        assert spy.done == []

    def test_record_transcribe_mode_runs_transcription(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved_wav = tmp_path / "out.wav"
        saved_note = tmp_path / "out.md"
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: saved_note,
        )

        spy = SpyCallbacks()
        recorder = FakeRecorder(finalized_path=saved_wav)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert spy.done == [saved_note]
        assert spy.record_only_done == []

    def test_finalize_failure_reports_error(self, tmp_path: Path):
        spy = SpyCallbacks()
        recorder = FakeRecorder(fail_finalize=True)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert spy.done == []
        assert spy.record_only_done == []
        assert any("エラー" in msg for msg in spy.errors)
        # 録音データがないと分かった時点で PortAudio の停止は試みない
        assert recorder.stopped is False

    def test_unexpected_exception_in_process_audio_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # 音声ファイル確定後、文字起こし以降で例外を起こし、
        # 外側の except (「予期せぬエラー」ハンドラ) を通ることを検証する
        errors: list[str] = []

        def fail_transcribe(*args, **kwargs):
            raise RuntimeError("想定外エラー")

        monkeypatch.setattr(workflow_module, "transcribe_and_save", fail_transcribe)

        callbacks = WorkflowCallbacks(
            on_status=lambda msg: None,
            on_log=lambda msg: None,
            on_recording_started=lambda: None,
            on_processing_started=lambda: None,
            on_done=lambda path: None,
            on_record_only_done=lambda path: None,
            on_error=errors.append,
        )

        recorder = FakeRecorder()
        log_file = tmp_path / "app.log"
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            callbacks,
            recorder_factory=lambda d, rec_dest: recorder,
            log_file=log_file,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert len(errors) == 1
        assert "文字起こしエラー" in errors[0]


class TestRecordingWorkflowTranscribeOnly:
    def test_runs_transcription_and_reports_done(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saved_note = tmp_path / "out.md"
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: saved_note,
        )

        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), thread_factory=ImmediateThread)

        wf.run_transcribe_only(tmp_path / "in.wav")

        assert spy.processing_started == 1
        assert spy.done == [saved_note]

    def test_transcription_error_is_reported(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        def fail(*args, **kwargs):
            raise RuntimeError("文字起こし失敗")

        monkeypatch.setattr(workflow_module, "transcribe_and_save", fail)

        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build(), thread_factory=ImmediateThread)

        wf.run_transcribe_only(tmp_path / "in.wav")

        assert spy.done == []
        assert any("文字起こしエラー" in msg for msg in spy.errors)


class TestRecordingWorkflowStopDoesNotBlockCaller:
    def test_stop_and_process_returns_immediately_while_stream_stop_blocks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """stream の停止が長時間ブロックしても、呼び出し元 (メインスレッド想定) は即座に戻る。"""
        block = threading.Event()
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
        )

        spy = SpyCallbacks()
        recorder = FakeRecorder(block_on_stop=block)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        # 実スレッドを使う (ImmediateThread ではブロックの意味がなくなるため)
        CapturingThread.instances = []
        wf._thread_factory = CapturingThread
        try:
            start = time.monotonic()
            wf.stop_and_process(MODE_RECORD_TRANSCRIBE)
            elapsed = time.monotonic() - start

            assert elapsed < 0.5
            assert wf.is_recording is False
        finally:
            block.set()
            for t in CapturingThread.instances:
                t.join(timeout=1.0)
                assert not t.is_alive()

    def test_start_rejected_while_stop_in_progress(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        block = threading.Event()
        spy = SpyCallbacks()
        recorder = FakeRecorder(block_on_stop=block)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        CapturingThread.instances = []
        wf._thread_factory = CapturingThread
        try:
            wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

            error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

            assert error is not None
            assert "停止" in error
        finally:
            block.set()
            for t in CapturingThread.instances:
                t.join(timeout=1.0)
                assert not t.is_alive()


class TestRecordingWorkflowStopTimeoutFallback:
    def test_stop_timeout_still_saves_recording(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """stop_with_timeout がタイムアウト (False) を返しても、既に確定済みの
        録音データはそのまま処理が続行される。"""
        saved_note = tmp_path / "out.md"
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: saved_note,
        )

        spy = SpyCallbacks()
        recorder = FakeRecorder(stop_timeout_result=False)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            stop_timeout=0.05,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert recorder.stop_timeout_used == 0.05
        assert spy.done == [saved_note]
        assert any("タイムアウト" in msg for msg in spy.logs)

    def test_finalize_happens_before_stop_with_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """finalize_recording() が stop_with_timeout() より先に呼ばれること
        (これが失われると、stop() のハングがファイル確定自体をブロックしてしまう)。"""
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
        )

        call_order: list[str] = []

        class OrderTrackingRecorder(FakeRecorder):
            def finalize_recording(self, timeout=2.0):
                call_order.append("finalize")
                return super().finalize_recording(timeout)

            def stop_with_timeout(self, timeout):
                call_order.append("stop_with_timeout")
                return super().stop_with_timeout(timeout)

        spy = SpyCallbacks()
        recorder = OrderTrackingRecorder()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert call_order == ["finalize", "stop_with_timeout"]

    def test_pending_close_blocks_restart_until_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """タイムアウトで放棄したストリームがまだ閉じ切っていない間は再録音を拒否する。"""
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
        )

        spy = SpyCallbacks()
        first_recorder = FakeRecorder(stop_timeout_result=False)
        first_recorder.is_closing = lambda: True  # 放棄後もまだクローズ処理中
        recorders = iter([first_recorder, FakeRecorder()])
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: next(recorders),
            stop_timeout=0.05,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        # start() が成功すると DeferredThread では捕捉できないタイマースレッドが
        # 再度起動するため、判定だけを行う 1 回目は ImmediateThread のまま呼ぶ
        # (拒否される想定なのでタイマースレッドは起動しない)
        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        assert error is not None
        assert "解放" in error

        first_recorder.is_closing = lambda: False
        wf._thread_factory = DeferredThread
        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)
        assert error is None
        assert wf._pending_close_recorder is None
        assert wf._recorder.started is True

    def test_pending_close_gives_up_after_grace_period_even_if_still_closing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """放棄したストリームが stop_timeout の3倍を超えても解放されない場合、
        is_closing() が True のままでも再録音を許可し、アプリが恒久的に
        使用不能になることを防ぐ (Issue #19 の失敗モードそのものへの対策)。"""
        monkeypatch.setattr(
            workflow_module,
            "transcribe_and_save",
            lambda audio_file, config, progress_callback=None: tmp_path / "out.md",
        )

        spy = SpyCallbacks()
        first_recorder = FakeRecorder(stop_timeout_result=False)
        first_recorder.is_closing = lambda: True  # 猶予期間を過ぎても閉じない
        recorders = iter([first_recorder, FakeRecorder()])
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: next(recorders),
            stop_timeout=0.05,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = ImmediateThread
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        # 猶予期間 (stop_timeout * 3 = 0.15秒) を過ぎたことにする
        wf._pending_close_since = time.monotonic() - 10.0

        wf._thread_factory = DeferredThread
        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        assert error is None
        assert wf._pending_close_recorder is None
        assert any("解放されない" in msg for msg in spy.logs)


class TestRecordingWorkflowStopThreadCreationFailure:
    def test_thread_creation_failure_does_not_stick_stopping_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """停止ワーカーのスレッド生成自体が失敗しても _stopping が固着せず、
        on_error が呼ばれ、以降の start() を永久に拒否し続けないこと。"""

        class FailingThreadFactory:
            def __init__(self, target, args=(), daemon=None):
                raise RuntimeError("スレッドを生成できません")

        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        wf._thread_factory = FailingThreadFactory
        wf.stop_and_process(MODE_RECORD_TRANSCRIBE)

        assert wf._stopping is False
        assert any("エラー" in msg for msg in spy.errors)

        wf._thread_factory = DeferredThread
        error = wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)
        assert error is None


class TestRecordingWorkflowShutdown:
    def test_shutdown_stops_recorder_and_clears_state(self, tmp_path: Path):
        spy = SpyCallbacks()
        recorder = FakeRecorder()
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        # shutdown() 内部の停止処理は実スレッドで動く (join(timeout) を要求するため)
        CapturingThread.instances = []
        wf._thread_factory = CapturingThread
        try:
            wf.shutdown()

            assert wf.is_recording is False
            assert recorder.finalize_called is True
            assert recorder.stopped is True
        finally:
            for t in CapturingThread.instances:
                t.join(timeout=1.0)
                assert not t.is_alive()

    def test_shutdown_does_not_block_when_stop_hangs(self, tmp_path: Path):
        block = threading.Event()
        spy = SpyCallbacks()
        recorder = FakeRecorder(block_on_stop=block)
        wf = RecordingWorkflow(
            VoiceNoteConfig(),
            spy.build(),
            recorder_factory=lambda d, rec_dest: recorder,
            thread_factory=DeferredThread,
        )
        wf.start(device_id=None, device_label="デバイスなし", rec_dest=tmp_path)

        CapturingThread.instances = []
        wf._thread_factory = CapturingThread
        try:
            start = time.monotonic()
            wf.shutdown()
            elapsed = time.monotonic() - start

            assert elapsed < 2.0
            assert wf.is_recording is False
        finally:
            block.set()
            for t in CapturingThread.instances:
                t.join(timeout=1.0)
                assert not t.is_alive()

    def test_shutdown_when_not_recording_is_noop(self):
        spy = SpyCallbacks()
        wf = RecordingWorkflow(VoiceNoteConfig(), spy.build())

        wf.shutdown()  # 例外を送出しないこと

        assert wf.is_recording is False
