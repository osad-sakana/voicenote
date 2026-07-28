"""recorder.ThreadedRecorder のユニットテスト。"""

import threading
import time

import numpy as np

from recorder import ThreadedRecorder


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
        finally:
            block.set()

    def test_data_survives_timeout(self):
        """タイムアウトしてストリームを放棄しても、既に取得済みの録音データは失われない。"""
        recorder = ThreadedRecorder()
        block = threading.Event()
        recorder._stream = FakeStream(block_event=block)
        recorder._running = True
        recorder._data = [np.zeros((16000, 1), dtype=np.float32)]

        try:
            result = recorder.stop_with_timeout(timeout=0.05)
            assert result is False
            data = recorder.get_data()
            assert len(data) == 16000
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
