"""recorder.ThreadedRecorder のユニットテスト。"""

import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

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


class FakeInputStream:
    """sd.InputStream の代わりに使うテスト用スタブ。

    実際の start() 経路を通して駆動できるよう、コンストラクタで受け取った
    callback をテスト側から手動で呼び出せるようにしている。
    """

    last_instance: "FakeInputStream | None" = None

    def __init__(self, samplerate, channels, dtype, device, callback):
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.last_instance = self

    def start(self):
        self.started = True
        # 実機では start() の直後からコールバックが発火しうる。ThreadedRecorder.start()
        # が _writer/_running の代入を stream.start() より前に済ませていないと、
        # このコールバックが黙って捨てられてしまう (回帰の検出用)。
        self.callback(np.zeros((160, 1), dtype=np.float32), 160, None, None)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FailingInputStream:
    last_instance: "FailingInputStream | None" = None

    def __init__(self, *a, **kw):
        self.closed = False
        FailingInputStream.last_instance = self

    def start(self):
        raise RuntimeError("デバイスが使用できません")

    def close(self):
        self.closed = True


class TestThreadedRecorderStartStop:
    def test_finalize_recording_returns_none_when_never_started(self):
        recorder = ThreadedRecorder()
        assert recorder.finalize_recording() is None

    def test_callback_writes_to_writer_via_real_start(self, tmp_path, monkeypatch):
        """recorder.start() を実際に呼び、start() が組み立てた _writer/_running
        の状態でコールバックが正しく書き込みへ届くことを検証する
        (start() 内の代入順序が壊れていないかの回帰チェック)。"""
        import recorder as recorder_module

        monkeypatch.setattr(recorder_module.sd, "InputStream", FakeInputStream)
        writer_holder: dict[str, FakeWriter] = {}

        def writer_factory(dest, rate):
            writer = FakeWriter(dest, rate)
            writer_holder["writer"] = writer
            return writer

        recorder = ThreadedRecorder(dest_dir=tmp_path, writer_factory=writer_factory)
        recorder.start()

        # FakeInputStream.start() が start() 呼び出し中に既に1回コールバックを
        # 発火させている。それが捨てられていなければ writer に届いているはず。
        assert len(writer_holder["writer"].frames) == 1

        chunk = np.zeros((160, 1), dtype=np.float32)
        FakeInputStream.last_instance.callback(chunk, 160, None, None)

        assert len(writer_holder["writer"].frames) == 2

    def test_callback_is_noop_after_finalize(self, tmp_path, monkeypatch):
        """finalize 後 (stop 完了前) にコールバックが発火しても例外を送出しない。"""
        import recorder as recorder_module

        monkeypatch.setattr(recorder_module.sd, "InputStream", FakeInputStream)
        recorder = ThreadedRecorder(
            dest_dir=tmp_path, writer_factory=lambda dest, rate: FakeWriter(dest, rate)
        )
        recorder.start()
        recorder.finalize_recording()

        chunk = np.zeros((160, 1), dtype=np.float32)
        FakeInputStream.last_instance.callback(chunk, 160, None, None)  # 例外を送出しないこと

    def test_start_aborts_writer_when_stream_start_fails(self, tmp_path, monkeypatch):
        import recorder as recorder_module

        monkeypatch.setattr(recorder_module.sd, "InputStream", FailingInputStream)

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
        # start() 失敗後は次の start() が再度使えるよう、状態が残っていないこと
        assert recorder._writer is None
        assert recorder._running is False
        # 生成済みのストリームハンドルがリークせずクローズされていること
        assert FailingInputStream.last_instance.closed is True

    def test_real_writer_produces_readable_wav_mid_recording(self, tmp_path, monkeypatch):
        """StreamingWavWriter を実際に使い、recorder.start() 経由のコールバックが
        finalize() を待たずに有効な WAV として読めることを確認する
        (受け入れ条件: stop() がハングしてもデータが失われない、の核心部分)。"""
        import recorder as recorder_module

        monkeypatch.setattr(recorder_module.sd, "InputStream", FakeInputStream)
        recorder = ThreadedRecorder(dest_dir=tmp_path)
        recorder.start()
        # FakeInputStream.start() が start() 呼び出し中に既に160フレーム分の
        # コールバックを1回発火させている。
        expected_frames = 160 + SAMPLE_RATE

        chunk = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        FakeInputStream.last_instance.callback(chunk, SAMPLE_RATE, None, None)

        writer = recorder._writer
        for _ in range(200):
            if writer.frames_written >= expected_frames:
                break
            time.sleep(0.01)

        with wave.open(str(writer.path), "rb") as f:
            assert f.getnframes() == expected_frames

        recorder.finalize_recording()


def _start_with_fake_stream(monkeypatch, tmp_path, chunk_sink=None, writer_factory=None):
    """FakeInputStream で recorder.start() を駆動するヘルパー。"""
    import recorder as recorder_module

    monkeypatch.setattr(recorder_module.sd, "InputStream", FakeInputStream)
    kwargs = {"dest_dir": tmp_path, "chunk_sink": chunk_sink}
    if writer_factory is not None:
        kwargs["writer_factory"] = writer_factory
    recorder = ThreadedRecorder(**kwargs)
    recorder.start()
    return recorder


def _fake_writer_factory(holder: dict):
    def factory(dest, rate):
        writer = FakeWriter(dest, rate)
        holder["writer"] = writer
        return writer

    return factory


class TestChunkSink:
    def test_sink_receives_copy_of_chunk(self, tmp_path, monkeypatch):
        """sink にはコピーが渡る (PortAudio のバッファ再利用で中身が変わらない)。"""
        received: list[np.ndarray] = []
        holder: dict = {}
        _start_with_fake_stream(
            monkeypatch, tmp_path, chunk_sink=received.append,
            writer_factory=_fake_writer_factory(holder),
        )

        chunk = np.full((160, 1), 0.5, dtype=np.float32)
        FakeInputStream.last_instance.callback(chunk, 160, None, None)
        chunk[:] = -1.0  # PortAudio によるバッファ再利用を模擬

        # start() 中の1回 + 手動の1回
        assert len(received) == 2
        assert received[1] is not chunk
        assert np.all(received[1] == 0.5)

    def test_sink_exception_does_not_affect_writer(self, tmp_path, monkeypatch):
        """sink が例外を出しても writer へ書かれるフレームは sink なしと同一。"""

        def failing_sink(chunk):
            raise RuntimeError("sink failure")

        chunks = [np.full((160, 1), i / 10, dtype=np.float32) for i in range(3)]

        def run(sink):
            holder: dict = {}
            _start_with_fake_stream(
                monkeypatch, tmp_path, chunk_sink=sink,
                writer_factory=_fake_writer_factory(holder),
            )
            for chunk in chunks:
                FakeInputStream.last_instance.callback(chunk, 160, None, None)
            return holder["writer"].frames

        with_sink = run(failing_sink)
        without_sink = run(None)

        assert len(with_sink) == len(without_sink) == 4
        for a, b in zip(with_sink, without_sink, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_sink_called_after_writer_write(self, tmp_path, monkeypatch):
        """writer.write() が先、sink が後に呼ばれる。"""
        order: list[str] = []
        holder: dict = {}

        class OrderedWriter(FakeWriter):
            def write(self, chunk):
                order.append("write")
                super().write(chunk)

        def factory(dest, rate):
            writer = OrderedWriter(dest, rate)
            holder["writer"] = writer
            return writer

        _start_with_fake_stream(
            monkeypatch, tmp_path, chunk_sink=lambda c: order.append("sink"),
            writer_factory=factory,
        )
        assert order == ["write", "sink"]

    def test_sink_still_called_when_writer_write_fails(self, tmp_path, monkeypatch):
        """writer の書き込み失敗は sink 呼び出しを妨げない (別 try/except)。"""
        received: list[np.ndarray] = []

        class FailingWriter(FakeWriter):
            def write(self, chunk):
                raise OSError("disk full")

        _start_with_fake_stream(
            monkeypatch, tmp_path, chunk_sink=received.append,
            writer_factory=lambda dest, rate: FailingWriter(dest, rate),
        )
        assert len(received) == 1

    def test_sink_not_called_after_finalize(self, tmp_path, monkeypatch):
        received: list[np.ndarray] = []
        recorder = _start_with_fake_stream(
            monkeypatch, tmp_path, chunk_sink=received.append,
            writer_factory=lambda dest, rate: FakeWriter(dest, rate),
        )
        recorder.finalize_recording()
        FakeInputStream.last_instance.callback(
            np.zeros((160, 1), dtype=np.float32), 160, None, None
        )
        assert len(received) == 1

    def test_no_sink_keeps_existing_behavior(self, tmp_path, monkeypatch):
        holder: dict = {}
        _start_with_fake_stream(
            monkeypatch, tmp_path, chunk_sink=None,
            writer_factory=_fake_writer_factory(holder),
        )
        FakeInputStream.last_instance.callback(
            np.zeros((160, 1), dtype=np.float32), 160, None, None
        )
        assert len(holder["writer"].frames) == 2

    def test_real_writer_output_identical_with_and_without_sink(self, tmp_path, monkeypatch):
        """実 StreamingWavWriter で sink あり/なしの WAV フレーム数・内容が一致する。"""
        rng = np.random.default_rng(0)
        chunks = [rng.uniform(-1, 1, (1600, 1)).astype(np.float32) for _ in range(5)]

        def run(subdir: str, sink):
            dest = tmp_path / subdir
            recorder = _start_with_fake_stream(monkeypatch, dest, chunk_sink=sink)
            for chunk in chunks:
                FakeInputStream.last_instance.callback(chunk, 1600, None, None)
            path = recorder.finalize_recording()
            with wave.open(str(path), "rb") as f:
                return f.getnframes(), f.readframes(f.getnframes())

        received: list[np.ndarray] = []
        frames_with, data_with = run("with", received.append)
        frames_without, data_without = run("without", None)

        assert frames_with == frames_without == 160 + 1600 * 5
        assert data_with == data_without
        assert len(received) == 6


class TestRecordAudioChunkSink:
    def _patch(self, monkeypatch, captured: dict):
        import recorder as recorder_module

        class SpyRecorder:
            def __init__(self, device_id=None, dest_dir=None, chunk_sink=None, **kwargs):
                captured["chunk_sink"] = chunk_sink
                captured["calls"] = []

            def start(self):
                captured["calls"].append("start")

            def finalize_recording(self, timeout=2.0):
                captured["calls"].append("finalize")
                return Path("/tmp/fake.wav")

            def stop_with_timeout(self, timeout=5.0):
                captured["calls"].append("stop")
                return True

        class ImmediateEvent:
            def set(self):
                pass

            def wait(self):
                return True

        monkeypatch.setattr(recorder_module, "ThreadedRecorder", SpyRecorder)
        monkeypatch.setattr(
            recorder_module, "threading", SimpleNamespace(Event=ImmediateEvent)
        )
        monkeypatch.setattr(recorder_module.signal, "signal", lambda *a, **kw: None)
        return recorder_module

    def test_record_audio_passes_chunk_sink(self, monkeypatch):
        captured: dict = {}
        recorder_module = self._patch(monkeypatch, captured)

        def sink(chunk):
            pass

        result = recorder_module.record_audio(chunk_sink=sink)

        assert result == Path("/tmp/fake.wav")
        assert captured["chunk_sink"] is sink
        # finalize_recording() → stop_with_timeout() の順序は維持
        assert captured["calls"] == ["start", "finalize", "stop"]

    def test_record_audio_default_sink_is_none(self, monkeypatch):
        captured: dict = {}
        recorder_module = self._patch(monkeypatch, captured)

        recorder_module.record_audio()

        assert captured["chunk_sink"] is None
