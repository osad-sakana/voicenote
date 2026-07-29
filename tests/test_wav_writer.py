"""wav_writer.StreamingWavWriter のユニットテスト。"""

import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from recorder import SAMPLE_RATE
from wav_writer import StreamingWavWriter, float32_to_int16


class TestFloat32ToInt16:
    def test_converts_float32_to_int16(self):
        # -1.0〜1.0 の float32 が int16 (-32768〜32767) に変換される
        audio = np.array([0.0, 0.5, 1.0, -1.0, -0.5], dtype=np.float32)
        data = float32_to_int16(audio)
        assert data.dtype == np.int16
        # 0.5 * 32767 ≈ 16383, -1.0 * 32767 = -32767
        assert data[0] == 0
        assert abs(int(data[1]) - 16383) <= 1
        assert int(data[2]) == 32767
        assert int(data[3]) == -32767


class TestStreamingWavWriter:
    def test_creates_dest_dir_if_missing(self, tmp_path: Path):
        target = tmp_path / "nested" / "out"
        writer = StreamingWavWriter(target, SAMPLE_RATE)
        writer.finalize()
        assert target.is_dir()

    def test_filename_matches_recording_pattern(self, tmp_path: Path):
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
        saved = writer.finalize()
        assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_recording\.wav$", saved.name)

    def test_accepts_string_dest_dir(self, tmp_path: Path):
        writer = StreamingWavWriter(str(tmp_path), SAMPLE_RATE)
        writer.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
        saved = writer.finalize()
        assert saved.exists()

    def test_file_is_valid_wav_before_finalize(self, tmp_path: Path):
        """finalize() を呼ぶ前でも、書き込み済みの分は有効なWAVとして読める
        (writeframes が呼び出しのたびにヘッダーを更新するため)。"""
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        chunk = np.linspace(-1.0, 1.0, SAMPLE_RATE, dtype=np.float32).reshape(-1, 1)
        writer.write(chunk)

        for _ in range(200):
            if writer.frames_written >= SAMPLE_RATE:
                break
            import time

            time.sleep(0.01)

        with wave.open(str(writer.path), "rb") as f:
            assert f.getnframes() == SAMPLE_RATE
            assert f.getframerate() == SAMPLE_RATE
            assert f.getsampwidth() == 2
            assert f.getnchannels() == 1

        writer.finalize()

    def test_preserves_sample_count(self, tmp_path: Path):
        n_samples = SAMPLE_RATE * 2  # 2秒分
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer.write(np.linspace(-1.0, 1.0, n_samples, dtype=np.float32).reshape(-1, 1))
        saved = writer.finalize()

        with wave.open(str(saved), "rb") as f:
            assert f.getnframes() == n_samples

    def test_zero_frames_returns_none_and_removes_file(self, tmp_path: Path):
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        path = writer.path

        result = writer.finalize()

        assert result is None
        assert not path.exists()

    def test_finalize_is_idempotent(self, tmp_path: Path):
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer.write(np.zeros(SAMPLE_RATE, dtype=np.float32))

        first = writer.finalize()
        second = writer.finalize()

        assert first == second

    def test_no_callback_after_finalize_raises(self, tmp_path: Path):
        """finalize 後に write() を呼んでも例外を送出しない
        (遅延コールバックがキューに積んでも、書き込みスレッドは既に終了済み)。"""
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer.finalize()

        writer.write(np.zeros(SAMPLE_RATE, dtype=np.float32))  # 例外を送出しないこと

    def test_unique_filenames_for_same_second(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import wav_writer

        fixed_timestamp = "2026-07-29_120000"
        monkeypatch.setattr(
            wav_writer,
            "datetime",
            type("_FixedDatetime", (), {"now": staticmethod(lambda: _FixedNow(fixed_timestamp))}),
        )

        writer1 = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer1.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
        writer1.finalize()

        writer2 = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        writer2.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
        path2 = writer2.finalize()

        assert path2.name != writer1.path.name
        assert "_2" in path2.stem

    def test_abort_removes_file_and_stops_thread(self, tmp_path: Path):
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        path = writer.path

        writer.abort()

        assert not path.exists()
        assert not writer._thread.is_alive()

    def test_finalize_timeout_does_not_close_handle_while_thread_busy(self, tmp_path: Path):
        """書き込みスレッドが writeframes() でまだ動いている間に finalize() が
        タイムアウトした場合、close() を呼んで seek と競合させてはならない。"""
        writer = StreamingWavWriter(tmp_path, SAMPLE_RATE)
        block = threading.Event()
        close_calls: list[bool] = []
        original_writeframes = writer._wav.writeframes
        original_close = writer._wav.close

        def blocking_writeframes(data):
            block.wait(timeout=5)
            original_writeframes(data)

        def tracking_close():
            close_calls.append(True)
            original_close()

        writer._wav.writeframes = blocking_writeframes
        writer._wav.close = tracking_close

        writer.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
        try:
            writer.finalize(timeout=0.05)
            assert close_calls == []
            assert writer._thread.is_alive()
        finally:
            block.set()
            writer._thread.join(timeout=2.0)

    def test_queue_full_drops_chunk_without_raising(self, tmp_path: Path, monkeypatch):
        """書き込みスレッドが詰まりキューが満杯でも write() は例外を送出せず、
        メモリ (キューサイズ) が録音長に比例して増え続けない。"""
        import wav_writer

        monkeypatch.setattr(wav_writer, "_MAX_QUEUE_SIZE", 1)
        writer = wav_writer.StreamingWavWriter(tmp_path, SAMPLE_RATE)

        block = threading.Event()
        original_writeframes = writer._wav.writeframes

        def blocking_writeframes(data):
            block.wait(timeout=5)
            original_writeframes(data)

        writer._wav.writeframes = blocking_writeframes

        try:
            writer.write(np.zeros(10, dtype=np.float32))
            for _ in range(100):  # スレッドがこの1件を取り出し writeframes でブロックするまで待つ
                if writer._queue.empty():
                    break
                time.sleep(0.01)
            writer.write(np.zeros(10, dtype=np.float32))  # キュー(maxsize=1)に積まれる
            writer.write(np.zeros(10, dtype=np.float32))  # キュー満杯 → 破棄されるだけで例外なし
        finally:
            block.set()
            writer.finalize(timeout=2.0)


class _FixedNow:
    def __init__(self, formatted: str):
        self._formatted = formatted

    def strftime(self, _fmt: str) -> str:
        return self._formatted
