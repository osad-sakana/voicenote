"""wav_writer.StreamingWavWriter のユニットテスト。"""

import re
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


class _FixedNow:
    def __init__(self, formatted: str):
        self._formatted = formatted

    def strftime(self, _fmt: str) -> str:
        return self._formatted
