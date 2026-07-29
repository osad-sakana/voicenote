"""recorder.ThreadedRecorder のユニットテスト。"""

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from recorder import SAMPLE_RATE, ThreadedRecorder


class FakeStream:
    def __init__(self, block_event: threading.Event | None = None, raise_on_stop=False):
        self._block_event = block_event
        self._raise_on_stop = raise_on_stop
        self.stop_called = False
        self.close_called = False

    def start(self):
        pass

    def stop(self):
        self.stop_called = True
        if self._raise_on_stop:
            raise RuntimeError("PortAudio エラー")
        if self._block_event is not None:
            self._block_event.wait()

    def close(self):
        self.close_called = True


class FakeWriter:
    """StreamingWavWriter の代わりに使うテスト用スタブ。"""

    def __init__(self, dest_dir=None, sample_rate=SAMPLE_RATE):
        self.path = Path("/tmp/fake_recording.wav")
        self.frames = []
        self.finalized = False
        self.aborted = False

    def write(self, chunk):
        self.frames.append(chunk)

    def finalize(self, timeout=2.0):
        self.finalized = True
        return self.path if self.frames else None

    def abort(self):
        self.aborted = True


class TestStopWithTimeout:
    def test_returns_true_and_clears_stream_when_stop_completes_in_time(self):
        recorder = ThreadedRecorder()
        stream = FakeStream()
        recorder._stream = stream
        recorder._running = True

        result = recorder.stop_with_timeout(timeout=1.0)

        assert result is True
        assert stream.stop_called is True
        assert stream.close_called is True
        assert recorder._stream is None
        assert recorder._running is False

    def test_returns_true_when_no_stream(self):
        recorder = ThreadedRecorder()

        result = recorder.stop_with_timeout(timeout=1.0)

        assert result is True

    def test_returns_false_when_stop_blocks_past_timeout(self):
        recorder = ThreadedRecorder()
        block = threading.Event()
        stream = FakeStream(block_event=block)
        recorder._stream = stream
        recorder._running = True

        try:
            start = time.monotonic()
            result = recorder.stop_with_timeout(timeout=0.05)
            elapsed = time.monotonic() - start

            assert result is False
            assert elapsed < 1.0
            # ストリームは即座に手放され、以降呼び出し元から再利用されない
            assert recorder._stream is None
            # 放棄した close 処理はバックグラウンドでまだ進行中のはず
            assert recorder.is_closing() is True
        finally:
            block.set()
            # block を解除すると、放棄されたクローズ処理もいずれ完了する
            for _ in range(100):
                if not recorder.is_closing():
                    break
                time.sleep(0.01)
            assert recorder.is_closing() is False

    def test_finalized_recording_survives_timeout(self):
        """タイムアウトしてストリームを放棄しても、既に確定済みの録音データは失われない。"""
        recorder = ThreadedRecorder()
        block = threading.Event()
        recorder._stream = FakeStream(block_event=block)
        recorder._running = True
        writer = FakeWriter()
        writer.write(np.zeros((16000, 1), dtype=np.float32))
        recorder._writer = writer

        audio_file = recorder.finalize_recording()
        assert audio_file == writer.path

        try:
            result = recorder.stop_with_timeout(timeout=0.05)
            assert result is False
            # finalize_recording は stop_with_timeout より前に完了しているため、
            # ここでも取得済みのパスは変わらない
            assert writer.finalized is True
        finally:
            block.set()

    def test_close_still_attempted_when_stop_raises(self):
        recorder = ThreadedRecorder()
        stream = FakeStream(raise_on_stop=True)
        recorder._stream = stream
        recorder._running = True

        # stop() 内で送出された例外は内部スレッドの中で発生するため、
        # 呼び出し元まで伝播しない (スレッドがハングしないことのみ保証する)
        result = recorder.stop_with_timeout(timeout=1.0)

        assert result is True
        assert stream.stop_called is True
        assert stream.close_called is True


class TestThreadedRecorderStartStop:
    def test_finalize_recording_returns_none_when_never_started(self):
        recorder = ThreadedRecorder()
        assert recorder.finalize_recording() is None

    def test_callback_writes_to_writer_while_running(self):
        recorder = ThreadedRecorder(writer_factory=lambda dest, rate: FakeWriter())
        recorder._writer = FakeWriter()
        recorder._running = True

        chunk = np.zeros((160, 1), dtype=np.float32)
        recorder._callback(chunk, 160, None, None)

        assert recorder._writer.frames == [chunk]

    def test_callback_is_noop_after_finalize(self):
        """finalize 後 (stop 完了前) にコールバックが発火しても例外を送出しない。"""
        recorder = ThreadedRecorder(writer_factory=lambda dest, rate: FakeWriter())
        recorder._writer = FakeWriter()
        recorder._running = True
        recorder.finalize_recording()

        chunk = np.zeros((160, 1), dtype=np.float32)
        recorder._callback(chunk, 160, None, None)  # 例外を送出しないこと

    def test_start_aborts_writer_when_stream_start_fails(self, tmp_path, monkeypatch):
        import recorder as recorder_module

        class FailingStream:
            def __init__(self, *a, **kw):
                pass

            def start(self):
                raise RuntimeError("デバイスが使用できません")

        monkeypatch.setattr(recorder_module.sd, "InputStream", FailingStream)

        aborted_writers = []

        def writer_factory(dest, rate):
            writer = FakeWriter(dest, rate)
            original_abort = writer.abort

            def abort():
                aborted_writers.append(writer)
                original_abort()

            writer.abort = abort
            return writer

        recorder = ThreadedRecorder(dest_dir=tmp_path, writer_factory=writer_factory)

        with pytest.raises(RuntimeError):
            recorder.start()

        assert len(aborted_writers) == 1
        assert aborted_writers[0].aborted is True

    def test_real_writer_produces_readable_wav_mid_recording(self, tmp_path):
        """StreamingWavWriter を実際に使い、start() 直後の書き込みが
        finalize() を待たずに有効な WAV として読めることを確認する
        (受け入れ条件: stop() がハングしてもデータが失われない、の核心部分)。"""
        recorder = ThreadedRecorder(dest_dir=tmp_path)
        recorder._writer = recorder._writer_factory(tmp_path, SAMPLE_RATE)
        recorder._running = True

        chunk = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        recorder._callback(chunk, SAMPLE_RATE, None, None)

        writer = recorder._writer
        for _ in range(200):
            if writer.frames_written >= SAMPLE_RATE:
                break
            time.sleep(0.01)

        with wave.open(str(writer.path), "rb") as f:
            assert f.getnframes() == SAMPLE_RATE

        recorder.finalize_recording()
